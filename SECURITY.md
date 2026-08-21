# Security policy

## Supported versions

The project is pre-1.0. Only the latest tagged release receives fixes; there are no
backports to older tags.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private vulnerability reporting:
[Report a vulnerability](https://github.com/epidemicsound/modal-workflows/security/advisories/new).

Include the affected version or commit, a description of the impact, and steps to
reproduce it. We aim to acknowledge reports within five working days and will keep you
updated while we investigate. Please give us a reasonable window to ship a fix before
any public disclosure.

## Scope

This library orchestrates Modal functions. Findings in Modal itself belong with
[Modal](https://modal.com/), not here. Relevant to this project are, for example:
handling of the `SLACK_WEBHOOK_URL` value, unsafe deserialization of workflow state
(step results are stored as pickles on a Modal Volume), and state isolation between
runs sharing a volume.

## A note on pickled state

Step results are persisted with `pickle` on the `workflow-state` Modal Volume.
Unpickling executes arbitrary code, so treat that volume as trusted storage: do not
point a workflow at a state volume or `state_name` that untrusted parties can write to.
