# mypy: ignore-errors
"""Integration test: deployed-app state isolation, retry stability, graceful restart.

Verifies that:
1. Concurrent calls to a deployed app each get an isolated state (different function_call_ids).
2. Modal retries of the same call preserve state (same function_call_id → same state_name,
   so step caches from the failed attempt are visible on retry).
3. Graceful restarts share state: two calls that carry the same _workflow_state_name
   kwarg resolve to the same state.

App lifecycle is managed entirely by this script.

Run from the repository root:
    .venv/bin/python tests/integration/test_deployed_app_isolation.py
"""

import sys

import modal

APP_NAME = "test-deployed-isolation"
APP_FILE = "tests/integration/deployed_isolation_app.py"


def _modal(*args: str) -> None:
    """Run a modal CLI subcommand using the same Python that is running this script."""
    import subprocess

    subprocess.run([sys.executable, "-m", "modal", *args], check=True)


def deploy() -> None:
    print(f"Deploying {APP_FILE}...")
    _modal("deploy", APP_FILE)
    print()


def stop() -> None:
    print(f"\nStopping app '{APP_NAME}'...")
    try:
        _modal("app", "stop", APP_NAME)
    except Exception as exc:
        print(f"  Warning: could not stop app: {exc}")


def test_concurrent_calls_get_different_state() -> None:
    """Three concurrent calls must each land in an isolated state."""
    print("Test 1: concurrent calls get distinct state names...")
    run_workflow = modal.Function.from_name(APP_NAME, "run_workflow")

    calls = [run_workflow.spawn(scenario="normal") for _ in range(3)]
    results = [c.get() for c in calls]
    state_names = [r["state_name"] for r in results]

    prefixes = {name.rsplit("_", 1)[0] for name in state_names}
    assert len(prefixes) == 1, f"Expected one app_id prefix across all calls, got: {prefixes}"

    assert len(set(state_names)) == 3, f"Expected 3 distinct state names but got: {state_names}"
    print(f"  PASS — state names: {state_names}")


def test_retry_preserves_state() -> None:
    """A workflow forced to fail on its first attempt must resume from cached state on retry.

    retry_workflow runs step_a, writes a marker, then raises. Modal retries the same
    function_call_id, so the state_name is unchanged. The retry finds step_a in cache
    (skips it) and finds the marker (skips the raise), then runs step_b and returns.
    If the retry used a *different* state_name the marker would be absent, the workflow
    would raise again, exhaust retries, and this call would raise here.
    """
    print("Test 2: Modal retry preserves state (step cached across retry boundary)...")
    retry_workflow = modal.Function.from_name(APP_NAME, "retry_workflow")

    result = retry_workflow.remote()

    assert result.get("retried") is True, f"Did not reach the retry path: {result}"
    assert result["step_a"] == "data:done", f"step_a wrong: {result}"
    assert result["step_b"] == "data:done:done", f"step_b wrong: {result}"
    print(f"  PASS — result: {result}")


def test_graceful_restart_kwarg_shares_state() -> None:
    """Two calls carrying the same _workflow_state_name must resolve to the same state.

    This is the mechanism used for graceful (self-managed timeout) restarts:
    the continuation is spawned with the same _workflow_state_name kwarg so it picks up
    where the original left off.
    """
    print("Test 3: explicit _workflow_state_name yields the same state name on both calls...")
    run_workflow = modal.Function.from_name(APP_NAME, "run_workflow")

    pinned_state = "integration-test-pinned-state"
    r1 = run_workflow.remote(scenario="normal", _workflow_state_name=pinned_state)
    r2 = run_workflow.remote(scenario="normal", _workflow_state_name=pinned_state)

    assert (
        r1["state_name"] == pinned_state
    ), f"Expected state_name to be '{pinned_state}', got: {r1['state_name']}"
    assert (
        r2["state_name"] == pinned_state
    ), f"Expected state_name to be '{pinned_state}', got: {r2['state_name']}"
    print(f"  PASS — shared state name: {r1['state_name']}")


if __name__ == "__main__":
    deploy()
    try:
        test_concurrent_calls_get_different_state()
        test_retry_preserves_state()
        test_graceful_restart_kwarg_shares_state()
        print("\nAll tests PASSED.")
    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        stop()
