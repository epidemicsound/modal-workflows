# mypy: ignore-errors
"""Alerting utilities for Modal functions.

Provides ``alert()`` for sending Slack messages and ``alert_on_error`` for
automatic exception alerting.  Both require ``SLACK_WEBHOOK_URL`` to be set
(typically via a Modal Secret).
"""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from typing import Any

import modal

logger = logging.getLogger(__name__)

SLACK_WEBHOOK_URL_ENV = "SLACK_WEBHOOK_URL"
MODAL_WORKSPACE_ENV = "MODAL_WORKSPACE"


def modal_app_url(app_id: str) -> str:
    """Build a Modal dashboard URL for the given app.

    The workspace slug is read from the ``MODAL_WORKSPACE`` environment variable;
    if it is unset the link omits the workspace segment.
    """

    environment = modal.config.config["environment"] or "dev"
    workspace = os.environ.get(MODAL_WORKSPACE_ENV)
    if workspace:
        return f"https://modal.com/apps/{workspace}/{environment}/{app_id}"
    return f"https://modal.com/apps/{environment}/{app_id}"


def _modal_context_lines(app_id: str) -> list[str]:
    """Return Slack-formatted context lines from the Modal runtime."""
    lines: list[str] = []

    try:
        fc_id = modal.current_function_call_id()
        if fc_id:
            lines.append(f">*Function call:* `{fc_id}`")
    except Exception:
        pass

    try:
        input_id = modal.current_input_id()
        if input_id:
            clean_id = input_id.split(":")[0]
            lines.append(f">*Input:* `{clean_id}`")
    except Exception:
        pass

    lines.append(f">*App:* <{modal_app_url(app_id)}|View in Modal>")
    return lines


def _send(text: str) -> None:
    """Send raw text to Slack. No extra context is appended."""
    from slack_sdk.webhook import WebhookClient

    webhook_url = os.environ.get(SLACK_WEBHOOK_URL_ENV)
    if not webhook_url:
        logger.warning(
            "Cannot send alert: %s environment variable is not set.", SLACK_WEBHOOK_URL_ENV
        )
        return

    client = WebhookClient(webhook_url)
    client.send(text=text)


def alert(message: str, app_id: str) -> None:
    """Send an alert message to Slack with Modal context automatically appended.

    Args:
        message (str): The alert text to send.
        app_id (str): Modal app ID (``ap-…``).
    """
    parts = [message, *_modal_context_lines(app_id)]
    _send("\n".join(parts))


def _format_error_alert(
    func: Callable[..., Any],
    exception: Exception,
    app_id: str,
) -> str:
    """Build a Slack-formatted error message."""
    name = f"{func.__module__}.{func.__qualname__}"
    error = f"{type(exception).__name__}: {exception}"

    lines = [
        f":rotating_light: *{name}* failed",
        f">`{error}`",
        *_modal_context_lines(app_id),
    ]

    return "\n".join(lines)


def send_exception_alert(
    func: Callable[..., Any],
    exception: Exception,
    app_id: str,
) -> None:
    """Format and send an error alert to Slack with Modal context.

    Args:
        func: The function that raised the exception.
        exception: The exception that was raised.
        app_id: Modal app ID (``ap-…``).
    """
    _send(_format_error_alert(func, exception, app_id=app_id))


def alert_on_error(app: modal.App) -> Callable[..., Any]:
    """Decorator that sends a Slack alert when the function raises.

    The original exception is always re-raised after the alert is sent.
    ``app.app_id`` is resolved at runtime so it is safe to pass an app
    that has not been deployed yet.

    Args:
        app: The ``modal.App`` instance.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                try:
                    send_exception_alert(func, e, app_id=app.app_id)
                except Exception as alert_err:
                    logger.exception(
                        "Failed to send alert for %s (original error: %s): %s",
                        func.__qualname__,
                        e,
                        alert_err,
                    )
                raise

        return wrapper

    return decorator
