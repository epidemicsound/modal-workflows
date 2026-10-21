# mypy: ignore-errors
from __future__ import annotations

import traceback
from typing import Any

WORKFLOW_ITEM_FAILURE_MARKER = "_workflow_item_failure"


def failure_from_exception(exc: BaseException) -> dict:
    """Plain-dict snapshot of an exception raised by a ``map()``/``parallel()`` item.

    A dict, not a custom class: caching the original exception isn't safe to pickle
    in general, and a custom class isn't readable by a caller without `modal_workflows`
    installed - same reason ``@workflow``'s restart marker is a dict too.
    """
    exc_type = type(exc)
    return {
        WORKFLOW_ITEM_FAILURE_MARKER: True,
        "exc_type": f"{exc_type.__module__}.{exc_type.__qualname__}",
        "message": str(exc),
        "traceback_str": "".join(traceback.format_exception(exc_type, exc, exc.__traceback__)),
    }


def is_workflow_item_failure(value: Any) -> bool:
    """True if *value* is a failure snapshot from ``ctx.map(..., return_exceptions=True)``."""
    return isinstance(value, dict) and bool(value.get(WORKFLOW_ITEM_FAILURE_MARKER, False))
