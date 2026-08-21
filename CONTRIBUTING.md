# Contributing to modal-workflows

Thanks for your interest in the project. Issues, discussions, and pull requests are
welcome.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

The project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/epidemicsound/modal-workflows.git
cd modal-workflows
make install        # uv sync --all-extras
make help           # list every target
```

That is everything you need for the unit tests and the lint gates. Running the
integration tests needs a Modal account as well, covered
[below](#integration-tests).

## Before you open a pull request

```bash
make lint    # black, isort, flake8, mypy
make test    # unit tests
```

Both run in CI against Python 3.10 to 3.13, so a green local run should mean a green
build. `make format` applies black and isort in place. `make help` lists every target,
grouped so the ones that cost money are obvious.

### Integration tests

`make test-integration` deploys and runs **real Modal apps**, which consumes Modal
credits and takes a long time. It is not part of CI. Run individual targets
(`make test-e2e`, `make test-training`, ...) when you touch the corresponding code, and
say in the pull request which ones you ran.

What you need:

1. **Authenticated Modal credentials.** Installing the CLI is not enough; run
   `modal token new`, or set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.
2. **A Modal environment matching `MODAL_ENVIRONMENT`**, which defaults to `dev`.
   Override it if your workspace uses a different name:

   ```bash
   make test-e2e MODAL_ENVIRONMENT=my-environment
   ```

3. **All extras installed locally** (`make install`). Every target does this for you.
   `make test-training` needs it specifically, because it imports the Lightning
   callback at module scope and so needs `torch` on the host, not just in the container.
4. **For `make test-alerts` only, a `slack-alerts-test` Modal secret** holding a
   `SLACK_WEBHOOK_URL`:

   ```bash
   modal secret create slack-alerts-test \
     SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   ```

No GPU is required: none of the integration tests request one, and the training test
forces its restarts with a short timeout on CPU.

Each target builds its container image from your working tree (see
`tests/integration/utilities.py`), so it tests your checkout rather than a released
version. Note `make test-deployed-isolation` runs `modal deploy` and leaves a deployed
app named `test-deployed-isolation` in the target environment.

## Conventions

- **Branches** are cut from `main`.
- **Commits and pull request titles** follow
  [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
  `<type>(<scope>): <subject>`, imperative mood, no trailing period. Example:
  `fix(workflows): keep step ids stable across restarts`. Pull requests are
  squash-merged with the title as the commit message.
- **Line length** is 100 characters, enforced by black and flake8.
- **Types**: `src/` is checked with `mypy --strict`. New code should be annotated.
  Existing modules that carry `# mypy: ignore-errors` are being migrated gradually;
  please do not add the pragma to new files.
- **Docstrings** use Google style with an `Args:`/`Returns:` section for public API.
- **Tests**: unit tests under `tests/workflows` and `tests/training` must not require
  network access or Modal credentials. Anything that needs a live Modal app belongs in
  `tests/integration`.

## Changing public behaviour

If a change affects the documented API, decorator arguments, environment variables, or
the Modal resources the library creates, update the README and add an entry to
`CHANGELOG.md` under `## Unreleased` in the same pull request.

## Releases

Maintainers only. The package is not published to PyPI; a release is a git tag plus a
GitHub Release with the built artifacts attached.

1. Bump `version` in `pyproject.toml`.
2. Move the `## Unreleased` entries in `CHANGELOG.md` under the new version heading.
3. Merge, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. The `Release` workflow verifies the tag matches `pyproject.toml`, builds the sdist
   and wheel, and creates the GitHub Release.
