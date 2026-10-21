# mypy: ignore-errors
"""In-memory stand-ins for the Modal Dict and Volume backing a ``WorkflowState``."""

from __future__ import annotations

import asyncio
import shutil
import threading
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock

import modal

import modal_workflows.workflows.state as state_module
from modal_workflows.workflows.context import WorkflowContext
from modal_workflows.workflows.state import WorkflowState


class ProcessKilled(BaseException):
    """Raised by a fake once its owning container has been killed."""


class DictNotFound(Exception):
    """Raised when deleting a Dict that does not exist without ``allow_missing``."""


class _DictMethod:
    """A Dict method that, like Modal's, can also be awaited through ``.aio``."""

    def __init__(self, call, delay_for=lambda *args: 0.0):
        self._call = call
        self._delay_for = delay_for

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        await asyncio.sleep(self._delay_for(*args))
        return self._call(*args, **kwargs)


class FakeModalDict:
    """Dict double that counts the requests made against it.

    Like a named Dict looked up with ``create_if_missing``, a deleted one comes back, empty,
    on the next request.
    """

    def __init__(self):
        self._data = {}
        self.dead = False
        self.deleted = False
        self.on_request = None
        self.requests = Counter()
        self.put_delays = {}
        self.clear_delay = 0.0
        self.put = _DictMethod(self.__setitem__, lambda key, *_: self.put_delays.get(key, 0.0))
        self.clear = _DictMethod(lambda: self._clear(), lambda: self.clear_delay)
        self._lock = threading.Lock()

    def _count(self, name):
        if self.dead:
            raise ProcessKilled()
        if self.on_request is not None:
            self.on_request(name)
        self.deleted = False
        with self._lock:
            self.requests[name] += 1

    def contains(self, key):
        self._count("contains")
        return key in self._data

    def get(self, key, default=None):
        self._count("get")
        return self._data.get(key, default)

    def __getitem__(self, key):
        self._count("get")
        return self._data[key]

    def __setitem__(self, key, value):
        self._count("put")
        self._data[key] = value

    def pop(self, key, default=None):
        self._count("pop")
        return self._data.pop(key, default)

    def items(self):
        self._count("items")
        return iter(list(self._data.items()))

    def _clear(self):
        self._count("clear")
        self._data.clear()

    @property
    def total_requests(self):
        return sum(self.requests.values())


class FakeModalVolume:
    """Volume double over a local directory that models commit durability.

    ``live`` is what the container sees. ``commit()`` snapshots it; ``crash()``
    rolls ``live`` back to the last snapshot, as if the container died with
    uncommitted writes. ``reload()`` is just as unforgiving: it drops uncommitted
    writes, so code must commit before reloading to keep them.
    """

    def __init__(self, live: Path, snapshot: Path):
        self._live = live
        self._snapshot = snapshot
        self._snapshot.mkdir(parents=True, exist_ok=True)
        self.commit_count = 0
        self.reload_count = 0
        self.events = []
        self.on_reload = None
        self.on_commit = None
        self.on_request = None
        self.dead = False

    def commit(self):
        if self.dead:
            raise ProcessKilled()
        if self.on_request is not None:
            self.on_request("commit")
        if self.on_commit is not None:
            self.on_commit()
        self.commit_count += 1
        self.events.append("commit")
        shutil.rmtree(self._snapshot)
        shutil.copytree(self._live, self._snapshot)

    def reload(self):
        if self.dead:
            raise ProcessKilled()
        if self.on_request is not None:
            self.on_request("reload")
        self.reload_count += 1
        self.events.append("reload")
        self.crash()
        if self.on_reload is not None:
            self.on_reload()

    def crash(self):
        shutil.rmtree(self._live)
        shutil.copytree(self._snapshot, self._live)


class Env:
    """A fake Modal backend shared by every context built from it."""

    def __init__(self, tmp_path: Path):
        self.live = tmp_path / "live"
        self.live.mkdir()
        self.dict = FakeModalDict()
        self.volume = FakeModalVolume(self.live, tmp_path / "snapshot")
        self.timeout = 3600

    def _delete_dict(self, name, allow_missing=False):
        if self.dict.deleted:
            if not allow_missing:
                raise DictNotFound(name)
            return
        self.dict._data.clear()
        self.dict.deleted = True

    def _create_or_get_volume(self, name):
        if name == state_module.workflow_volumes.WORKFLOW_STATE_VOLUME_NAME:
            return self.volume
        return MagicMock()

    def new_state(self) -> WorkflowState:
        return WorkflowState("run")

    def new_context(self) -> WorkflowContext:
        ctx = WorkflowContext(
            app=MagicMock(),
            func=MagicMock(),
            func_name="wf",
            state_name="run",
            args=(),
            kwargs={},
            timeout=self.timeout,
        )
        self.volume.reload_count = 0
        self.volume.commit_count = 0
        self.volume.events.clear()
        return ctx

    def kill(self) -> None:
        """Kills the running container: every further Modal request raises ProcessKilled."""
        self.volume.dead = True
        self.dict.dead = True

    def restart_after_kill(self, lose_uncommitted: bool = True) -> WorkflowContext:
        """Starts a fresh container's context, optionally losing uncommitted volume writes."""
        self.volume.dead = False
        self.dict.dead = False
        if lose_uncommitted:
            self.volume.crash()
        return self.new_context()

    def trace_requests(self, kill_before: int | None = None) -> list[str]:
        """Records every Dict and Volume request; kills the container before request number
        *kill_before* (counting from zero) takes effect."""
        trace: list[str] = []
        lock = threading.Lock()

        def on_request(name: str) -> None:
            with lock:
                position = len(trace)
                trace.append(name)
            if position == kill_before:
                self.kill()
                raise ProcessKilled()

        self.dict.on_request = on_request
        self.volume.on_request = on_request
        return trace

    def reset(self) -> None:
        """Returns the backend to an empty, alive state."""
        self.volume.dead = False
        self.dict.dead = False
        self.dict.on_request = None
        self.volume.on_request = None
        self.dict._data.clear()
        for directory in (self.live, self.volume._snapshot):
            shutil.rmtree(directory)
            directory.mkdir()


def make_function(results=None, fail=(), fail_with=RuntimeError):
    """A marked modal function double whose ``spawn(item)`` returns a finished call.

    Items in *fail* raise ``fail_with(...)`` when their call is read.
    """
    spawned = []
    calls = {}

    def spawn(item, *args, **kwargs):
        spawned.append(item)
        call = MagicMock()
        calls[f"fc-{item}"] = call
        call.object_id = f"fc-{item}"
        if item in fail:
            call.get.side_effect = fail_with(f"boom {item}")
        else:
            call.get.return_value = item * 10 if results is None else results[item]
        return call

    func = MagicMock(spec=modal.Function)
    func.spawn.side_effect = spawn
    func.spawned = spawned
    func.calls = calls
    return func
