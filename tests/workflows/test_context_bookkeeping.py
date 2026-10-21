# mypy: ignore-errors
"""Bookkeeping behaviour of ``map``/``parallel`` against a real ``WorkflowState`` on fakes."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import modal_workflows.workflows.context as context_module
from modal_workflows.workflows.failures import is_workflow_item_failure

from tests.workflows.fakes import ProcessKilled, make_function  # isort: skip


class VirtualClock:
    """Replaces wall-clock waiting: ``sleep`` advances ``now`` instead of blocking."""

    def __init__(self):
        self.now = 0.0

    def sleep(self, seconds):
        self.now += seconds

    def time(self):
        return self.now


@pytest.fixture
def clock():
    virtual = VirtualClock()
    with patch.object(time, "sleep", virtual.sleep), patch.object(time, "time", virtual.time):
        yield virtual


def trigger_restart(ctx):
    """Makes the workflow believe its time budget is used up."""
    ctx.start_time = float("-inf")


def finished_call(call_id, value):
    call = MagicMock()
    call.object_id = call_id
    call.get.return_value = value
    return call


def call_ids(ctx):
    """Step ids that currently have a persisted function-call id."""
    return set(ctx.state.cached_function_call_ids())


def reattach(calls):
    """Makes Modal hand back the given calls (by id) when a continuation re-attaches."""
    return patch.object(
        context_module.modal.FunctionCall, "from_id", side_effect=lambda i: calls[i]
    )


class TestResume:
    def test_results_committed_by_another_container_are_seen_at_map_start(self, env):
        ctx = env.new_context()
        other = env.new_state()

        def other_container_commits_item_3():
            env.volume.on_reload = None
            other.cache_step_result("m_3", "from-elsewhere")

        env.volume.on_reload = other_container_commits_item_3
        func = make_function()

        results = ctx.map(func, list(range(5)), "m")

        assert results[3] == "from-elsewhere"
        assert 3 not in func.spawned

    def test_resumed_map_skips_cached_items_and_spawns_the_rest(self, env):
        first = env.new_context()
        first.state.cache_step_result("m_0", "r0")
        first.state.cache_step_result("m_2", "r2")
        second = env.new_context()
        func = make_function()

        results = second.map(func, [0, 1, 2, 3], "m")

        assert results == ["r0", 10, "r2", 30]
        assert sorted(func.spawned) == [1, 3]

    def test_resumed_map_reruns_an_item_whose_cached_result_file_is_truncated(self, env):
        first = env.new_context()
        first.state.cache_step_result("m_0", "r0")
        first.state.cache_step_result("m_1", "r1")
        result_path = first.state.step_results_directory / "m_1.pkl"
        result_path.write_bytes(result_path.read_bytes()[:-2])
        first.state.commit()
        func = make_function()

        results = env.new_context().map(func, [0, 1], "m")

        assert results == ["r0", 10]
        assert func.spawned == [1]

    def test_resume_inherits_in_flight_calls_by_id_without_respawning(self, env):
        ctx = env.new_context()
        ctx.state.cache_function_call_id("m_0", "fc-old-0")
        ctx.state.cache_function_call_id("m_1", "fc-old-1")
        restored = {f"fc-old-{i}": finished_call(f"fc-old-{i}", f"result-{i}") for i in (0, 1)}
        func = make_function()

        with reattach(restored):
            results = ctx.map(func, ["a", "b"], "m")

        assert results == ["result-0", "result-1"]
        func.spawn.assert_not_called()

    def test_a_cached_result_with_a_stale_call_id_clears_the_id(self, env):
        ctx = env.new_context()
        ctx.state.cache_step_result("m_0", "done")
        ctx.state.cache_function_call_id("m_0", "fc-stale")

        results = ctx.map(make_function(), [0], "m")

        assert results == ["done"]
        assert call_ids(ctx) == set()

    def test_every_spawned_call_id_is_cleared_once_its_result_is_cached(self, env):
        ctx = env.new_context()

        ctx.map(make_function(), list(range(4)), "m")

        assert call_ids(ctx) == set()


def synced_requests(env, item_count):
    """Dict reads, volume reloads and commits of a map that resumes half-done, with failures."""
    env.reset()
    earlier = env.new_context()
    for i in range(0, item_count, 2):
        earlier.state.cache_step_result(f"m_{i}", i * 10)
        earlier.state.cache_function_call_id(f"m_{i}", f"fc-{i}")
    ctx = env.new_context()
    env.dict.requests.clear()
    failing = {i for i in range(item_count) if i % 5 == 1}

    ctx.map(make_function(fail=failing), list(range(item_count)), "m", return_exceptions=True)

    dict_reads = sum(env.dict.requests[name] for name in ("items", "get", "contains"))
    return dict_reads, env.volume.reload_count, env.volume.commit_count


class TestRequestCosts:
    def test_state_reads_reloads_and_commits_do_not_grow_with_the_number_of_items(self, env):
        small = synced_requests(env, 5)
        large = synced_requests(env, 50)

        assert all(big <= little for big, little in zip(large, small)), (small, large)

    def test_clearing_the_call_ids_of_a_finished_map_does_not_grow_with_the_number_of_items(
        self, env
    ):
        def cleanup_requests(item_count):
            env.reset()
            ctx = env.new_context()
            env.dict.requests.clear()
            ctx.map(make_function(), list(range(item_count)), "m")
            return env.dict.requests["pop"] + env.dict.requests["clear"]

        assert cleanup_requests(50) <= cleanup_requests(5)


class TestCommitAtEndOfRun:
    @pytest.mark.parametrize("other_ids_cached", [False, True])
    def test_a_failing_id_cleanup_leaves_results_durable_and_resumable(self, env, other_ids_cached):
        ctx = env.new_context()
        if other_ids_cached:
            ctx.state.cache_function_call_id("elsewhere", "fc-elsewhere")

        def unavailable(*args, **kwargs):
            raise RuntimeError("dict unavailable")

        with (
            patch.object(env.dict, "pop", unavailable),
            patch.object(env.dict, "_clear", unavailable),
        ):
            with pytest.raises(RuntimeError, match="dict unavailable"):
                ctx.map(make_function(), list(range(3)), "m")

        env.volume.crash()
        restarted = env.new_context()
        func = make_function()
        assert restarted.map(func, list(range(3)), "m") == [0, 10, 20]
        func.spawn.assert_not_called()

    def test_a_stuck_id_cleanup_does_not_stall_the_map_and_leaves_results_resumable(self, env):
        ctx = env.new_context()
        env.dict.clear_delay = 60

        with patch.object(context_module, "CALL_ID_CLEAR_TIMEOUT_SECONDS", 0.05):
            assert ctx.map(make_function(), list(range(3)), "m") == [0, 10, 20]

        env.dict.clear_delay = 0
        env.volume.crash()
        func = make_function()
        assert env.new_context().map(func, list(range(3)), "m") == [0, 10, 20]
        func.spawn.assert_not_called()

    def test_all_results_are_durable_when_the_map_returns(self, env):
        ctx = env.new_context()

        ctx.map(make_function(), list(range(7)), "m")
        env.volume.crash()

        durable = env.new_state().completed_steps
        assert durable == {f"m_{i}" for i in range(7)}

    def test_a_kill_before_the_end_keeps_call_ids_so_a_continuation_refetches_the_results(
        self, env
    ):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def killing_get(**_):
            env.kill()
            raise ProcessKilled()

        def spawn(item):
            call = inner_spawn(item)
            if item == 5:
                call.get.side_effect = killing_get
            return call

        func.spawn.side_effect = spawn

        with pytest.raises(ProcessKilled):
            ctx.map(func, list(range(10)), "m")

        restarted = env.restart_after_kill()
        assert env.new_state().completed_steps == set()
        assert call_ids(restarted) == {f"m_{i}" for i in range(10)}
        restored = {f"fc-{i}": finished_call(f"fc-{i}", i * 10) for i in range(10)}
        second_run = make_function()
        with reattach(restored):
            results = restarted.map(second_run, list(range(10)), "m")

        assert results == [i * 10 for i in range(10)]
        second_run.spawn.assert_not_called()
        assert call_ids(restarted) == set()

    def test_after_an_aborting_failure_a_continuation_reruns_only_the_failed_item(self, env):
        ctx = env.new_context()
        func = make_function(fail={3})

        with pytest.raises(RuntimeError, match="boom 3"):
            ctx.map(func, list(range(6)), "m")

        continuation = env.restart_after_kill()
        fixed = make_function()
        with patch.object(
            context_module.modal.FunctionCall, "from_id", side_effect=func.calls.__getitem__
        ):
            results = continuation.map(fixed, list(range(6)), "m")

        assert results == [i * 10 for i in range(6)]
        assert fixed.spawned == [3]

    def test_a_call_id_from_outside_the_map_survives_it_and_is_re_attached_after_a_restart(
        self, env
    ):
        ctx = env.new_context()
        side = make_function()
        ctx.spawn(side, "side", 7)

        ctx.map(make_function(), list(range(4)), "m")

        env.volume.crash()
        continuation = env.new_context()
        with patch.object(
            context_module.modal.FunctionCall, "from_id", side_effect=side.calls.__getitem__
        ):
            assert continuation.spawn(side, "side", 7).get() == 70
        assert side.spawned == [7]
        assert call_ids(continuation) == set()

    def test_a_restart_makes_finished_results_durable_before_the_continuation_is_spawned(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        durable_at_handoff = []

        def spawn(item):
            call = inner_spawn(item)
            if item == 2:
                call.get.side_effect = lambda **_: trigger_restart(ctx) or 20
            if item >= 3:
                call.get.side_effect = TimeoutError()
            return call

        def continuation_spawn(*args, **kwargs):
            env.volume.crash()
            durable_at_handoff.append(env.new_state().completed_steps)

        func.spawn.side_effect = spawn
        ctx.func.spawn.side_effect = continuation_spawn

        with pytest.raises(context_module.WorkflowRestarted):
            ctx.map(func, list(range(5)), "m")

        assert {"m_0", "m_1", "m_2"} <= durable_at_handoff[0]
        assert call_ids(ctx) == {f"m_{i}" for i in range(5)}

    def test_each_item_runs_once_across_a_restart_and_its_continuation(self, env, clock):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        calls = {}

        def spawn(item):
            calls[item] = inner_spawn(item)
            if item == 2:
                value = calls[item].get.return_value

                def get(**_):
                    if "polled" not in calls:
                        calls["polled"] = True
                        trigger_restart(ctx)
                        raise TimeoutError()
                    return value

                calls[item].get.side_effect = get
            return calls[item]

        func.spawn.side_effect = spawn

        with pytest.raises(context_module.WorkflowRestarted):
            ctx.map(func, list(range(4)), "m")

        env.volume.crash()
        continuation = env.new_context()
        by_id = {f"fc-{i}": calls[i] for i in range(4) if i in calls}
        with reattach(by_id):
            results = continuation.map(func, list(range(4)), "m")

        assert results == [0, 10, 20, 30]
        assert sorted(func.spawned) == [0, 1, 2, 3]
        assert call_ids(continuation) == set()


class TestConcurrentSpawn:
    def test_spawns_run_concurrently_up_to_the_parallelism_limit(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        lock = threading.Lock()
        running = {"now": 0, "max": 0}
        rendezvous = threading.Barrier(4, timeout=5)

        def spawn(item):
            with lock:
                running["now"] += 1
                running["max"] = max(running["max"], running["now"])
            if item < 4:
                rendezvous.wait()
            with lock:
                running["now"] -= 1
            return inner_spawn(item)

        func.spawn.side_effect = spawn

        with patch.object(context_module, "MAX_PARALLEL_REQUESTS", 8):
            ctx.map(func, list(range(40)), "m")

        assert 4 <= running["max"] <= 8

    def test_results_keep_index_order_when_spawns_finish_out_of_order(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            time.sleep(0.02 * (5 - item))
            return inner_spawn(item)

        func.spawn.side_effect = spawn

        results = ctx.map(func, list(range(6)), "m")

        assert results == [i * 10 for i in range(6)]

    def test_one_slow_spawn_does_not_hold_back_the_others(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        others_done = threading.Event()

        def spawn(item):
            if item == 0:
                assert others_done.wait(timeout=5), "the other spawns waited for item 0"
            call = inner_spawn(item)
            if len(func.spawned) == 39:
                others_done.set()
            return call

        func.spawn.side_effect = spawn

        with patch.object(context_module, "MAX_PARALLEL_REQUESTS", 2):
            results = ctx.map(func, list(range(40)), "m")

        assert results == [i * 10 for i in range(40)]

    def test_a_stuck_call_id_write_does_not_stall_the_map(self, env):
        ctx = env.new_context()
        env.dict.put_delays[ctx.state._function_call_id_key("m_3")] = 60

        with patch.object(context_module, "CALL_ID_WRITE_TIMEOUT_SECONDS", 0.05):
            results = ctx.map(make_function(), list(range(8)), "m")

        assert results == [i * 10 for i in range(8)]

    def test_a_call_id_whose_write_timed_out_is_saved_before_a_restart(self, env):
        ctx = env.new_context()
        env.dict.put_delays[ctx.state._function_call_id_key("m_1")] = 60
        func = make_function()
        inner_spawn = func.spawn.side_effect
        ids_when_continuation_spawned = []

        def spawn(item):
            call = inner_spawn(item)
            if item == 2:
                call.get.side_effect = restart_on_first_poll
            return call

        def restart_on_first_poll(**_):
            if not ids_when_continuation_spawned:
                trigger_restart(ctx)
                raise TimeoutError()
            return 20

        func.spawn.side_effect = spawn
        ctx.func.spawn.side_effect = lambda *a, **k: ids_when_continuation_spawned.append(
            call_ids(ctx)
        )

        with patch.object(context_module, "CALL_ID_WRITE_TIMEOUT_SECONDS", 0.05):
            with pytest.raises(context_module.WorkflowRestarted):
                ctx.map(func, list(range(4)), "m")

        assert ids_when_continuation_spawned == [{f"m_{i}" for i in range(4)}]
        env.volume.crash()
        continuation = env.new_context()
        with patch.object(
            context_module.modal.FunctionCall, "from_id", side_effect=func.calls.__getitem__
        ):
            results = continuation.map(func, list(range(4)), "m")
        assert results == [0, 10, 20, 30]
        assert sorted(func.spawned) == [0, 1, 2, 3]

    def test_each_spawned_call_id_is_persisted_before_its_result_is_collected(self, env):
        ctx = env.new_context()
        func = make_function()
        persisted_when_collected = {}
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            value = call.get.return_value

            def get(**_):
                persisted_when_collected[item] = ctx.state.get_cached_function_call_id(f"m_{item}")
                return value

            call.get.side_effect = get
            return call

        func.spawn.side_effect = spawn

        ctx.map(func, list(range(5)), "m")

        assert persisted_when_collected == {i: (True, f"fc-{i}") for i in range(5)}

    def test_a_spawn_failure_propagates(self, env):
        ctx = env.new_context()
        func = make_function()
        func.spawn.side_effect = RuntimeError("quota exceeded")

        with pytest.raises(RuntimeError, match="quota exceeded"):
            ctx.map(func, [0, 1, 2], "m")

    def test_a_spawn_failure_stops_spawning_and_keeps_the_ids_of_spawns_already_running(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        started = []

        def spawn(item):
            started.append(item)
            if item == 0:
                raise RuntimeError("quota exceeded")
            time.sleep(0.3)
            return inner_spawn(item)

        func.spawn.side_effect = spawn

        with patch.object(context_module, "MAX_PARALLEL_REQUESTS", 2):
            with pytest.raises(RuntimeError, match="quota exceeded"):
                ctx.map(func, list(range(8)), "m")

        assert 3 not in started
        running = {item for item in started if item != 0}
        assert call_ids(ctx) == {f"m_{i}" for i in running}

    def test_a_restart_waits_for_in_flight_spawns_and_the_continuation_spawns_the_rest(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect
        ids_when_continuation_spawned = []

        def spawn(item):
            if item == 10:
                trigger_restart(ctx)
            time.sleep(0.01)
            return inner_spawn(item)

        def continuation_spawn(*args, **kwargs):
            ids_when_continuation_spawned.append(call_ids(ctx))

        func.spawn.side_effect = spawn
        ctx.func.spawn.side_effect = continuation_spawn

        with pytest.raises(context_module.WorkflowRestarted):
            ctx.map(func, list(range(200)), "m")

        spawned_before_restart = list(func.spawned)
        assert 10 <= len(spawned_before_restart) < 200
        assert ids_when_continuation_spawned == [{f"m_{i}" for i in spawned_before_restart}]

        env.volume.crash()
        continuation = env.new_context()
        with patch.object(
            context_module.modal.FunctionCall, "from_id", side_effect=func.calls.__getitem__
        ):
            results = continuation.map(func, list(range(200)), "m")

        assert results == [i * 10 for i in range(200)]
        assert sorted(func.spawned) == list(range(200))

    def test_a_kill_between_spawn_and_id_persist_respawns_only_items_without_an_id(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 3:
                env.kill()
            return call

        func.spawn.side_effect = spawn

        with pytest.raises(ProcessKilled):
            ctx.map(func, list(range(8)), "m")

        restarted = env.restart_after_kill()
        persisted = {int(step_id.split("_")[1]) for step_id in call_ids(restarted)}
        restored = {f"fc-{i}": finished_call(f"fc-{i}", i * 10) for i in range(8)}
        second_run = make_function()
        with reattach(restored):
            results = restarted.map(second_run, list(range(8)), "m")

        assert results == [i * 10 for i in range(8)]
        assert 3 not in persisted
        assert set(second_run.spawned) == set(range(8)) - persisted


class TestCollection:
    def test_results_keep_index_order_when_earlier_calls_finish_last(self, env, clock):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 0:
                call.get.side_effect = [TimeoutError(), TimeoutError(), 0]
            return call

        func.spawn.side_effect = spawn

        results = ctx.map(func, list(range(5)), "m")

        assert results == [0, 10, 20, 30, 40]

    def test_an_earlier_failure_is_raised_before_a_later_one_and_only_it_is_cancelled(self, env):
        ctx = env.new_context()
        func = make_function(fail={1, 4})

        with pytest.raises(RuntimeError, match="boom 1"):
            ctx.map(func, list(range(6)), "m")

        func.calls["fc-1"].cancel.assert_called_once()
        func.calls["fc-4"].cancel.assert_not_called()

    def test_an_earlier_inherited_failure_is_raised_before_a_later_spawned_one(self, env):
        ctx = env.new_context()
        ctx.state.cache_function_call_id("m_0", "fc-old-0")
        inherited = MagicMock()
        inherited.get.side_effect = RuntimeError("boom inherited 0")
        func = make_function(fail={1})

        with reattach({"fc-old-0": inherited}):
            with pytest.raises(RuntimeError, match="boom inherited 0"):
                ctx.map(func, [0, 1], "m")

        inherited.cancel.assert_called_once()
        func.calls["fc-1"].cancel.assert_not_called()

    def test_a_polling_error_on_a_call_ahead_of_its_turn_is_not_its_outcome(self, env, clock):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 0:
                call.get.side_effect = [TimeoutError(), 0]
            if item == 1:
                call.get.side_effect = [ConnectionError("network blip"), 10]
            return call

        func.spawn.side_effect = spawn

        results = ctx.map(func, [0, 1, 2], "m", return_exceptions=True)

        assert results == [0, 10, 20]

    def test_a_persistent_failure_ahead_of_its_turn_is_still_cached_as_a_failure(self, env, clock):
        ctx = env.new_context()
        func = make_function(fail={1})
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 0:
                call.get.side_effect = [TimeoutError(), 0]
            return call

        func.spawn.side_effect = spawn

        results = ctx.map(func, [0, 1, 2], "m", return_exceptions=True)

        assert is_workflow_item_failure(results[1])
        assert results[0] == 0 and results[2] == 20
        restarted = env.new_context()
        again = make_function()
        assert is_workflow_item_failure(restarted.map(again, [0, 1, 2], "m")[1])
        again.spawn.assert_not_called()

    def test_polls_made_ahead_of_an_unfinished_call_do_not_grow_with_the_number_of_items(
        self, env, clock
    ):
        def polls_ahead(item_count):
            env.reset()
            clock.now = 0.0
            ctx = env.new_context()
            func = make_function()
            inner_spawn = func.spawn.side_effect
            ahead = []

            def spawn(item):
                call = inner_spawn(item)
                value = call.get.return_value

                def get(**_):
                    if clock.now < 2:
                        if item:
                            ahead.append(item)
                        raise TimeoutError()
                    return value

                call.get.side_effect = get
                return call

            func.spawn.side_effect = spawn
            ctx.map(func, list(range(item_count)), "m")
            return len(ahead)

        assert polls_ahead(1000) <= polls_ahead(100)

    def test_wait_timeout_counts_from_the_items_turn_and_is_cached_as_a_failure(self, env, clock):
        ctx = env.new_context()
        finishes_at = {0: 60.0, 1: 140.0}
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            value = call.get.return_value

            def get(**_):
                if clock.now < finishes_at.get(item, float("inf")):
                    raise TimeoutError()
                return value

            call.get.side_effect = get
            return call

        func.spawn.side_effect = spawn

        results = ctx.map(func, [0, 1, 2], "m", wait_timeout=100, return_exceptions=True)

        assert results[0] == 0
        assert results[1] == 10, "item 1 finished within 100 s of its turn, though not of the map"
        assert is_workflow_item_failure(results[2])
        assert results[2]["message"] == "Function call did not complete before timeout."
        func.calls["fc-2"].cancel.assert_called_once()
        assert ctx.state.read_step_result("m_2")[0] is True

    def test_restart_during_an_idle_wait_commits_finished_results_first(self, env, clock):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 2:

                def get(**_):
                    trigger_restart(ctx)
                    raise TimeoutError()

                call.get.side_effect = get
            return call

        func.spawn.side_effect = spawn

        with pytest.raises(context_module.WorkflowRestarted):
            ctx.map(func, list(range(4)), "m")

        env.volume.crash()
        assert env.new_state().completed_steps == {"m_0", "m_1"}


class TestPollingAUnfinishedCall:
    @pytest.mark.parametrize("kind", ["map", "step"])
    def test_a_call_unfinished_for_hours_is_polled_sparsely_and_returned_promptly(
        self, env, clock, kind
    ):
        env.timeout = 10**9
        ctx = env.new_context()
        finishes_at = 6 * 3600
        polls = []
        call = MagicMock()
        call.object_id = "fc-1"

        def get(**_):
            polls.append(clock.now)
            if clock.now < finishes_at:
                raise TimeoutError()
            return "done"

        call.get.side_effect = get
        func = make_function()
        func.spawn.side_effect = lambda *a, **k: call

        if kind == "map":
            result = ctx.map(func, [0], "m")[0]
        else:
            result = ctx.step(func, "s")

        assert result == "done"
        assert clock.now - finishes_at <= 10
        assert len(polls) <= finishes_at / 10 + 20


def run_map(ctx, func, mode):
    items = list(range(8))
    if mode == "parallel":
        results = ctx.parallel([(func, i) for i in items], "m", return_exceptions=True)
    else:
        results = ctx.map(func, items, "m", return_exceptions=True)
    return [r["message"] if is_workflow_item_failure(r) else r for r in results]


class TestCrashAtEveryRequest:
    @pytest.mark.parametrize("mode", ["map", "parallel"])
    @pytest.mark.parametrize("lose_uncommitted", [True, False])
    def test_a_continuation_after_a_kill_at_any_request_finishes_the_run_exactly_once(
        self, env, mode, lose_uncommitted
    ):
        clean_func = make_function(fail={2, 5})
        trace = env.trace_requests()
        expected = run_map(env.new_context(), clean_func, mode)
        total_requests = len(trace)
        last_id_put = max(i for i, name in enumerate(trace) if name == "put")
        assert "commit" in trace[last_id_put + 1 : -1]

        for kill_before in range(total_requests):
            env.reset()
            func = make_function(fail={2, 5})
            env.trace_requests(kill_before)
            try:
                run_map(env.new_context(), func, mode)
            except ProcessKilled:
                pass  # The simulated kill is expected; the continuation below must recover.
            env.dict.on_request = env.volume.on_request = None

            continuation = env.restart_after_kill(lose_uncommitted)
            with patch.object(
                context_module.modal.FunctionCall, "from_id", side_effect=func.calls.__getitem__
            ):
                results = run_map(continuation, func, mode)

            where = f"kill before request {kill_before}"
            assert results == expected, where
            assert env.dict._data == {}, where
            assert set(func.spawned) == set(range(8)), where
            if kill_before > last_id_put:
                assert sorted(func.spawned) == list(range(8)), where


class TestFinalCommitFailures:
    def test_a_failing_final_commit_surfaces_and_keeps_every_call_id(self, env):
        ctx = env.new_context()
        func = make_function()

        def failing_commit():
            raise RuntimeError("commit unavailable")

        env.volume.on_commit = failing_commit

        with pytest.raises(RuntimeError, match="commit unavailable"):
            ctx.map(func, list(range(4)), "m")

        assert call_ids(ctx) == {f"m_{i}" for i in range(4)}
        env.volume.on_commit = None
        restarted = env.restart_after_kill()
        with patch.object(
            context_module.modal.FunctionCall, "from_id", side_effect=func.calls.__getitem__
        ):
            results = restarted.map(func, list(range(4)), "m")
        assert results == [0, 10, 20, 30]
        assert sorted(func.spawned) == [0, 1, 2, 3]

    def test_a_failing_restart_commit_spawns_no_continuation_and_keeps_the_ids(self, env):
        ctx = env.new_context()
        func = make_function()
        inner_spawn = func.spawn.side_effect

        def spawn(item):
            call = inner_spawn(item)
            if item == 2:
                call.get.side_effect = lambda **_: trigger_restart(ctx) or 20
            if item >= 3:
                call.get.side_effect = TimeoutError()
            return call

        func.spawn.side_effect = spawn

        def failing_commit():
            raise RuntimeError("commit unavailable")

        env.volume.on_commit = failing_commit

        with pytest.raises(RuntimeError, match="commit unavailable"):
            ctx.map(func, list(range(5)), "m")

        ctx.func.spawn.assert_not_called()
        assert {"m_0", "m_1", "m_2"} <= call_ids(ctx)


class TestFinish:
    def test_finish_deletes_the_state_dict_and_every_result(self, env):
        ctx = env.new_context()
        ctx.map(make_function(), list(range(3)), "m")
        ctx.state.cache_function_call_id("leftover", "fc-leftover")

        ctx.finish()

        assert env.dict.deleted
        env.volume.crash()
        assert env.new_state().completed_steps == set()

    def test_finish_succeeds_when_the_state_dict_is_already_gone(self, env):
        ctx = env.new_context()
        ctx.map(make_function(), list(range(3)), "m")
        env._delete_dict("state")

        ctx.finish()

        assert env.dict.deleted
        assert env.new_state().completed_steps == set()

    def test_a_run_under_the_same_state_after_finish_starts_fresh(self, env):
        first = env.new_context()
        first.map(make_function(), list(range(3)), "m")
        first.finish()
        func = make_function()

        results = env.new_context().map(func, list(range(3)), "m")

        assert results == [0, 10, 20]
        assert sorted(func.spawned) == [0, 1, 2]


class TestCallsThatAreGone:
    @pytest.mark.parametrize("error_name", ["OutputExpiredError", "NotFoundError"])
    def test_a_re_attached_call_that_is_gone_is_spawned_again(self, env, error_name):
        ctx = env.new_context()
        ctx.state.cache_function_call_id("m_0", "fc-old-0")
        expired = MagicMock()
        expired.get.side_effect = getattr(context_module.modal.exception, error_name)("gone")
        func = make_function()

        with reattach({"fc-old-0": expired}):
            results = ctx.map(func, [0, 1], "m", return_exceptions=True)

        assert results == [0, 10]
        assert sorted(func.spawned) == [0, 1]
        assert call_ids(ctx) == set()

    @pytest.mark.parametrize("error_name", ["OutputExpiredError", "NotFoundError"])
    def test_an_item_whose_step_raises_the_error_is_spawned_twice_then_propagates(
        self, env, error_name
    ):
        ctx = env.new_context()
        error = getattr(context_module.modal.exception, error_name)
        func = bounded_respawns(make_function(fail={1}, fail_with=error))

        with pytest.raises(error):
            ctx.map(func, [0, 1, 2], "m")

        assert func.spawned.count(1) == 2

    @pytest.mark.parametrize("error_name", ["OutputExpiredError", "NotFoundError"])
    def test_with_return_exceptions_the_item_is_a_failure_and_others_are_unaffected(
        self, env, error_name
    ):
        ctx = env.new_context()
        error = getattr(context_module.modal.exception, error_name)
        func = bounded_respawns(make_function(fail={1}, fail_with=error))

        results = ctx.map(func, [0, 1, 2], "m", return_exceptions=True)

        assert results[0] == 0
        assert is_workflow_item_failure(results[1])
        assert results[2] == 20
        assert func.spawned.count(1) == 2
        assert call_ids(ctx) == set()


def bounded_respawns(func, limit=5):
    """Fails the test, instead of hanging it, if one item is spawned more than *limit* times."""
    inner_spawn = func.spawn.side_effect

    def spawn(item, *args, **kwargs):
        if func.spawned.count(item) >= limit:
            pytest.fail(f"item {item} was spawned more than {limit} times")
        return inner_spawn(item, *args, **kwargs)

    func.spawn.side_effect = spawn
    return func
