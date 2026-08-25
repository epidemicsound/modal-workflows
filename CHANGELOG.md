# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the version is below
1.0.0, minor releases may contain breaking changes.

The package is not published to PyPI. Each release is a git tag with the built sdist and
wheel attached to the corresponding [GitHub Release](https://github.com/epidemicsound/modal-workflows/releases).

## Unreleased

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
