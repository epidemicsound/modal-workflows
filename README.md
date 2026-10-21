# modal-workflows

[![CI](https://github.com/epidemicsound/modal-workflows/actions/workflows/ci.yml/badge.svg)](https://github.com/epidemicsound/modal-workflows/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Python utilities for running resumable ML workflows and long training jobs on
[Modal](https://modal.com/), with optional Slack alerting.

Modal enforces a hard 24-hour wall-clock cap per function call. Both decorators in this
package work around it the same way: they checkpoint progress to a Modal Volume and hand
off to a fresh container before the cap is reached, so a run that takes days survives as
a chain of containers instead of dying at hour 24.

## What's included?

| Area | Import path | Role |
|------|-------------|------|
| Workflows | `modal_workflows.workflows` | `@workflow`, `WorkflowContext`, restarts, Slack alerts |
| Training | `modal_workflows.training` | `@training` (framework-agnostic restart handoff) and `TrainingCheckpointRestartCallback` (Lightning) for checkpoint + timeout handoff |

Optional dependency groups in `pyproject.toml`: `training`, `dev`.

## `@workflow` or `@training`?

Both survive Modal's 24-hour wall-clock cap by checkpointing and gracefully handing
off to a fresh container (`self_managed_timeout`, 23h by default), and both share the
same state volume, `state_name` resolution, and Slack alerting. The difference is the
shape of the work:

- **`@workflow`** for a **multi-step pipeline**. The decorated function receives a
  `WorkflowContext` and breaks the work into steps (`ctx.step`, `ctx.map`) that each run
  as their own Modal function. Step results are cached, so after a timeout or crash the
  workflow resumes from the last completed step instead of re-running finished ones. Reach
  for it when you have a DAG of distinct stages, for example preprocess, train, evaluate,
  or fan-out/fan-in work.

- **`@training`** for a **single long-running training job** that runs in one Modal
  function. It manages checkpoint + timeout handoff: before the cap is hit the run
  checkpoints and continues in a new container from that checkpoint. Use it for one big
  training loop that outlives the container limit. The Lightning
  `TrainingCheckpointRestartCallback` wires the checkpointing in automatically; the
  handoff itself is framework-agnostic. It also covers multi-node clusters (`cluster_size`,
  GPU, RDMA).

Rule of thumb: orchestrating **several steps** goes to `@workflow`; resuming **one long
train loop** goes to `@training`.

## Requirements

- Python 3.10 to 3.13
- A [Modal](https://modal.com/) account with local credentials (`pip install modal && modal token new`)
- Git, if you install from this repository with a `git+` URL (see below)

## Installation

This package is not published to PyPI. Install it from this repository, pinned to a
release tag:

```bash
pip install "modal-workflows @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0"

# with the Lightning training callback (pulls in torch)
pip install "modal-workflows[training] @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0"
```

With uv:

```bash
uv add "modal-workflows @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0"
```

In a `requirements.txt`:

```
modal-workflows @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0
```

Pin a tag rather than a branch. The project is pre-1.0 and minor versions may break
compatibility. Every release also attaches a built wheel and sdist to its
[GitHub Release](https://github.com/epidemicsound/modal-workflows/releases) if you prefer
to vendor the artifact.

### Installing it into a Modal image

Your Modal container image needs the package too, both for the orchestrator function and
for every step function. `git+` URLs require git in the image:

```python
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "modal-workflows @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0"
    )
)
```

To avoid installing git, point pip at the tag's source archive instead:

```python
image = modal.Image.debian_slim().pip_install(
    "https://github.com/epidemicsound/modal-workflows/archive/refs/tags/v0.2.0.tar.gz"
)
```

Add the `training` extra (`modal-workflows[training] @ git+...`) only where you need the
Lightning callback.

### Local development

Install from a checkout in editable mode:

```bash
# with uv
uv sync --all-extras

# or with pip
pip install -e ".[dev,training]"
```

The `-e` flag installs the package in editable mode, so changes to the source are
immediately available without reinstalling.

## Quickstart

```python
import modal

from modal_workflows.workflows import workflow

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install(
        "modal-workflows @ git+https://github.com/epidemicsound/modal-workflows@v0.2.0"
    )
)
app = modal.App("example", image=image)


@app.function()
def add(a: int, b: int) -> int:
    return a + b


@app.function()
def double(x: int) -> int:
    return x * 2


@workflow(app)
def pipeline(ctx):
    total = ctx.step(add, "add", 3, 4)
    doubled = ctx.map(double, [1, 2, 3], step_id="double")
    ctx.finish()
    return {"sum": total, "doubled": doubled}


@app.local_entrypoint()
def main():
    print(pipeline.remote())
```

Any `modal.Function` can be a step: a function decorated with `@app.function()`, or a
bound method of an `@app.cls` class such as `MyModel().predict`. Anything else, such as a
plain callable or the class itself, makes `ctx.step`, `ctx.spawn`, `ctx.map`, and
`ctx.parallel` raise `TypeError`. `@workflow` and `@training` functions are entry points,
not steps: call a `@training` function directly rather than nesting it in a workflow.
Classes using `@modal.enter`, `@modal.concurrent`, or a different image from the workflow
are supported. A class with a slow `@modal.enter` should set `startup_timeout` explicitly,
because `timeout` is also the default startup limit.

`retries` may be passed to `@workflow` and `@training` (unlike `timeout`): a graceful
restart returns a value rather than raising, so Modal never treats it as a failure to
retry.
`ctx.finish()` clears the persisted state to mark the run as complete, so call it only
on the success path.

## How resumption works

Each step gets a `step_id` that must be unique within a run. Results are pickled to the
state volume under that id, and in-flight async calls record their Modal function-call
id. When a container is replaced, whether from the self-managed timeout, a crash, or a
preemption, the new container reads the same state: completed steps return their cached
result, and steps still running are inherited by call id instead of being started again.

When `@workflow` hands off at its self-managed timeout, the call returns
`{"_workflow_restarted": True, "call_id": "<id>"}` instead of the function's normal
result. It is a plain dict so that a caller without this package installed can read it.
The continuation is already running by then; wait for it with
`modal.FunctionCall.from_id(call_id).get()`. Calling the workflow again yourself does not
resume it. Without an explicit `state_name` that starts a new, unrelated run, and with the
same explicit `state_name` it races the continuation on the same state.

State is keyed by `state_name`:

- **Omitted** (the default): derived as `{app.app_id}_{function_call_id}`. Each
  invocation gets isolated state, while Modal retries of the same call share it.
- **Explicit**: used as-is. Pass the same `state_name` to deliberately resume a specific
  earlier run, for example after a crash you have since fixed.

`WorkflowState` hashes the name to a fixed 32-character identifier before naming its Modal
Dict and step-result directory after it, so it stays within Modal's 64-character object-name limit.
The mapping from `state_name` to the hashed name is logged at startup.

```python
@workflow(app, state_name="nightly-retrain")
def nightly(ctx): ...
```

Both `state_name` and `step_id` are used in directory and file names on the state volume,
so they are validated as single path components: up to 128 characters, starting with a
letter or digit and containing only letters, digits, `.`, `_` and `-`. Anything else,
including path separators and `..`, raises `ValueError`. `ctx.map()` and `ctx.parallel()`
key each item as `{step_id}_{index}`, so that full key must fit within the limit; it is
checked before anything is spawned.

### Modal resources it creates

All are created with `create_if_missing=True` the first time they are used, in whichever
Modal environment your credentials point at.

| Resource | Where | Contents |
|----------|-------|----------|
| Volume `workflow-state` (v2) | mounted at `/mnt/workflow-state` | Pickled step results (under the hashed state name), training checkpoints and restart metadata (under `<state_name>/`) |
| Dict `state-<hashed state_name>` | Modal Dict | Function-call ids of in-flight steps, for restart inheritance |

Only the `@workflow` orchestrator and `@training` functions mount the state volume; step
functions do not. Outputs such as model files belong on storage you mount on your own
functions. Because step results are unpickled from `workflow-state`, treat that volume as
trusted storage (see [SECURITY.md](SECURITY.md)).

## `WorkflowContext` API

| Member | Purpose |
|--------|---------|
| `ctx.step(func, step_id, *args, wait_timeout=None, **kwargs)` | Run one step synchronously and cache its result |
| `ctx.spawn(func, step_id, *args, **kwargs)` | Start a step asynchronously; the returned handle's `.get()` stays timeout-aware |
| `ctx.map(func, items, step_id, *args, wait_timeout=None, return_exceptions=False, **kwargs)` | Fan out over `items`, caching per item as `{step_id}_{index}` |
| `ctx.parallel([(func, *args), ...], step_id, wait_timeout=None, return_exceptions=False)` | Run different steps concurrently, caching each individually |
| `ctx.finish()` | Clear persisted state after a successful run |
| `ctx.should_restart` / `ctx.restart_if_needed()` | Check the self-managed timeout, or hand off now if it has elapsed |
| `ctx.log_event(message)` | Log a workflow event through Python's `logging` module |

The step helpers already check the timeout between steps. Call `ctx.restart_if_needed()`
yourself inside long stretches of local work in the orchestrator.

### Caching and ordering in `ctx.map`

`ctx.map` caches each result by the item's position in `items`. When a workflow resumes,
it runs the workflow body again from the top and rebuilds the list, so `items` must come
back in the same order. Sort lists whose order can vary between runs, such as a query
without `ORDER BY`, a directory or bucket listing, or a `set`. Otherwise a cached result
is silently attributed to the wrong item.

### Handling failed items

By default, the first failing item in `ctx.map` or `ctx.parallel` raises and aborts the
workflow. With `return_exceptions=True`, the slot for a failed item holds a plain-dict
snapshot of the exception instead (keys `exc_type`, `message`, `traceback_str`), and the
other items carry on:

```python
from modal_workflows.workflows import is_workflow_item_failure

results = ctx.map(process, items, step_id="process", return_exceptions=True)
failed = [r for r in results if is_workflow_item_failure(r)]
```

Failures are cached like results, so a resumed run does not retry them. A graceful restart
is never treated as an item failure. The snapshot is a dict rather than the exception
object so that it survives pickling and can be read by a caller that does not have this
package installed.

## Long training runs

```python
import pytorch_lightning as pl

from modal_workflows.training import training
from modal_workflows.training.callbacks import TrainingCheckpointRestartCallback

SELF_MANAGED_TIMEOUT_SECONDS = 23 * 3600


@training(app, gpu="A100", self_managed_timeout=SELF_MANAGED_TIMEOUT_SECONDS)
def train():
    callback = TrainingCheckpointRestartCallback(
        save_interval_seconds=15 * 60,
        self_managed_timeout_seconds=SELF_MANAGED_TIMEOUT_SECONDS,
    )
    trainer = pl.Trainer(max_epochs=100, callbacks=[callback])
    trainer.fit(model, datamodule=datamodule, ckpt_path=callback.resume_checkpoint_path())
```

Pass the same value to the decorator and the callback so the checkpoint and the handoff
line up. When the timeout elapses the callback saves a checkpoint, writes restart
metadata, and stops training cleanly; `@training` then spawns the continuation and
returns a `WorkflowRestartedResult`, so Modal records the container as succeeding.
`callback.resume_checkpoint_path()` returns `None` on the first run and the checkpoint
path on every continuation.

Without Lightning, write your own checkpoint loop and keep the same contract: save state
under `ctx`-independent paths on the state volume and return normally before the cap.

### Multi-node

```python
@training(app, gpu="H100:8", cluster_size=4, rdma=True)
def train_cluster(): ...
```

`cluster_size > 1` requires a `gpu` argument and routes through
`modal.experimental.clustered`. Only rank 0 performs the restart handoff and sends
alerts. `DistributedLaunchContext.modal_multi_node(cluster_size, gpu_per_node)` builds
the rank and master-address context for the current container, and `torchrun_argv()`
turns it into a `torch.distributed.run` command line.

## Configuration

| Environment variable | Required | Effect |
|----------------------|----------|--------|
| `SLACK_WEBHOOK_URL` | No | Enables `alert()` and `alert_on_error`. Without it, alerting logs a warning and continues. Supply it through a Modal Secret. |
| `MODAL_WORKSPACE` | No | Your Modal workspace slug, so dashboard links in alerts point at the right workspace. |
| `MODAL_ENVIRONMENT` | No | Standard Modal variable. Also used to build dashboard links; omitted from the link when unset. |
| `MODAL_WORKFLOWS_TRAINING_STATE_NAME` | Set for you | Written by `@training` so the Lightning callback can find the checkpoint directory. Do not set it yourself. |

## Slack alerting

Alerting is optional. With `SLACK_WEBHOOK_URL` set, `@workflow` and `@training` post a
message when the decorated function raises (`alert_on_error=True` by default), with the
function call id, input id, and a dashboard link attached. `alert(message, app_id)` sends
your own message, and the `alert_on_error(app)` decorator adds the same behaviour to any
Modal function. `slack_sdk` must be installed in the container image; it is a dependency
of this package, so installing `modal-workflows` in the image is enough.

## Development

```bash
make help      # list every target, grouped
make install   # uv sync --all-extras
make test      # unit tests
make lint      # black, isort, flake8, mypy
make format    # apply black and isort
```

`make` on its own prints the same list as `make help`. Unit tests need no Modal
credentials and no network access.

### Running the integration tests

The integration targets deploy and run **real Modal apps**, so they cost money and take
time. They are not part of CI. `make help` lists them under their own heading; run
`make test-integration` for all of them, or a single target such as `make test-e2e`.

They need:

- **Authenticated Modal credentials** (`modal token new`, or `MODAL_TOKEN_ID` and
  `MODAL_TOKEN_SECRET`). Having the CLI installed is not sufficient on its own.
- **`MODAL_TESTING_ENVIRONMENT` set to the Modal environment to run in.** There is no
  default, and the targets stop if it is unset. Use a scratch environment rather than one
  that runs real workflows, because the tests write to the shared `workflow-state` volume
  there:

  ```bash
  modal environment create workflows-testing   # once, any name works
  export MODAL_TESTING_ENVIRONMENT=workflows-testing
  make test-e2e
  ```

- **A `slack-alerts-test` Modal secret, for `make test-alerts` only**, holding the
  webhook the test posts to:

  ```bash
  modal secret create slack-alerts-test \
    SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
  ```

No GPU is needed for any of them. Each target builds its image from your working tree,
so it exercises your checkout rather than a released version. `make test-deployed-isolation`
additionally runs `modal deploy` and leaves a deployed app behind in the target
environment.

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions and the release process.

## Versioning and stability

The project follows [Semantic Versioning](https://semver.org/) but is still pre-1.0:
minor releases may change public behaviour. Pin an exact tag and read
[CHANGELOG.md](CHANGELOG.md) before upgrading. `@training` uses
`modal.experimental.clustered` for multi-node runs, which is experimental in Modal
itself.

## Security

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for attribution.
