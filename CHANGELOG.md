# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is below
1.0.0, minor releases may contain breaking changes.

The package is not published to PyPI. Each release is a git tag with the built sdist and
wheel attached to the corresponding [GitHub Release](https://github.com/epidemicsound/modal-workflows/releases).

## Unreleased

## 0.2.0

### Breaking

- Removed the `workflow-artifacts` volume, together with `@workflow_function`,
  `ctx.artifacts_run_path`, `ctx.artifacts_shared_path`, `ctx.commit_artifacts_volume()`
  and `ctx.dashboard_url`. Any `modal.Function` can now be a step. Write outputs to
  storage you mount on your own functions.
- Step functions no longer mount any workflow volume. Only the `@workflow` orchestrator
  and `@training` functions mount `workflow-state`.
- `@workflow` returns `{"_workflow_restarted": True, "call_id": "<id>"}` on a graceful
  restart instead of a `WorkflowRestartedResult`. `@training` still returns
  `WorkflowRestartedResult`.
- `WorkflowState` hashes `state_name` to a fixed-length identifier for its Modal Dict and
  step-result directory. Runs started on 0.1.x will not find their cached step results
  after upgrading, so finish in-flight workflows before you upgrade.
- `ctx.log_event()` logs through Python's `logging` module instead of appending to an
  event-log file on the artifacts volume.
- `@training` no longer sets the `MODAL_WORKFLOWS_TRAINING_TIMEOUT_SECONDS` environment
  variable.
- Requires `modal>=1.3.2`.

### Added

- `return_exceptions=True` on `ctx.map()` and `ctx.parallel()` stores a failed item as a
  plain-dict snapshot instead of aborting the run. The snapshot is cached, so a resumed
  run skips the item. Detect snapshots with
  `modal_workflows.workflows.is_workflow_item_failure`.
- Bound methods of `@app.cls` classes can be used as workflow steps.

### Changed

- The integration test targets no longer assume a Modal environment name. Set
  `MODAL_TESTING_ENVIRONMENT` to the environment they should run in; the targets stop if
  it is unset.
- `modal_app_url()` leaves the environment out of the link when no Modal environment is
  configured, instead of assuming a default name.
- Faster orchestration. `ctx.map()` and `ctx.parallel()` write results without committing
  them and commit the state volume once at the end. They poll in-flight calls with
  backoff, and clear function-call ids concurrently after the commit.
- `ctx.map()` and `ctx.parallel()` spawn their items concurrently from a thread pool, so
  items are no longer submitted to Modal in strict index order. Results still come back
  in index order. A hard crash of the orchestrator (for example an out-of-memory kill)
  while items are being spawned can now re-run up to 16 items on resume instead of one,
  plus any whose call-id write was still pending, so keep step functions idempotent.
- A call id whose write to the state Dict takes longer than 30 seconds no longer blocks
  spawning. It is kept in memory and saved again before a graceful restart. When a batch
  owns every cached call id, the ids are cleared in a single request.
- `ctx.finish()` deletes the run's function-call-id Dict instead of only emptying it.
- After a restart, a re-attached call whose output has expired, or that Modal no longer
  knows about, is spawned again instead of failing the step.
- State names longer than Modal's 64-character object-name limit are supported, up to
  the 128-character identifier limit. The mapping from `state_name` to the hashed name
  is logged.

## 0.1.1

First public release.

### Added

- `modal_workflows.workflows`: `@workflow` and `@workflow_function` decorators,
  `WorkflowContext` (`step`, `spawn`, `map`, `parallel`, `finish`), step-result caching
  and restart handoff before Modal's 24-hour wall-clock cap, artifact and state volumes,
  and optional Slack alerting.
- `modal_workflows.training`: `@training` decorator for checkpoint and timeout handoff
  of a single long-running job, multi-node cluster support (`cluster_size`, `rdma`),
  `DistributedLaunchContext` for torchrun, and the PyTorch Lightning
  `TrainingCheckpointRestartCallback`.
- `py.typed` marker, so type information is visible to consumers of the package.
- Continuous integration (lint, type-check, unit tests on Python 3.10 to 3.13, and a
  clean-environment install check of the built wheel).
- Release workflow that builds the sdist and wheel and attaches them to a GitHub Release
  on tag push.
- Contribution, security, and code of conduct documentation; issue and pull request
  templates; Dependabot configuration.
- `make lint` and `make format` targets, and a `flake8` configuration aligned with
  black's 100-character line length.
- `MODAL_ENVIRONMENT` is overridable in the integration test targets instead of being
  hardcoded to `dev`.
