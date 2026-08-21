# mypy: ignore-errors
"""Validation for identifiers that are used as filesystem path components.

``state_name`` and ``step_id`` are caller-supplied and are joined onto volume
mount paths to build directory and file names. Without validation, values such
as ``"../../etc"`` or ``"/etc/passwd"`` would resolve outside the intended
directory (``Path("/mnt/x") / "/etc/passwd"`` is ``/etc/passwd``), so every
identifier must be a single, well-formed path component.
"""

from __future__ import annotations

import re

MAX_IDENTIFIER_LENGTH = 128

_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def validate_path_identifier(value: str, kind: str) -> str:
    """Validates that *value* is safe to use as a single path component.

    Only ASCII letters, digits, ``.``, ``_`` and ``-`` are allowed, and the first
    character must be a letter or digit. That rejects path separators, ``..``,
    absolute paths, leading dots and null bytes, so a validated identifier can
    never escape the directory it is joined to.

    Args:
        value (str): The identifier to validate.
        kind (str): Identifier name used in error messages, e.g. ``"state_name"``.

    Returns:
        str: The identifier, unchanged.

    Raises:
        ValueError: If the identifier is not a string, is empty, exceeds
            ``MAX_IDENTIFIER_LENGTH``, or contains disallowed characters.
    """
    if not isinstance(value, str):
        raise ValueError(f"{kind} must be a string, got '{type(value).__name__}'.")
    if not value:
        raise ValueError(f"{kind} must not be empty.")
    if len(value) > MAX_IDENTIFIER_LENGTH:
        raise ValueError(
            f"{kind} must be at most {MAX_IDENTIFIER_LENGTH} characters, got {len(value)}.",
        )
    if not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(
            f"Invalid {kind} '{value}'. It is used as a directory or file name, so it must "
            "start with a letter or digit and contain only letters, digits, '.', '_' and '-'.",
        )
    return value


def validate_state_name(state_name: str) -> str:
    """Validates a workflow state name. See :func:`validate_path_identifier`."""
    return validate_path_identifier(state_name, "state_name")


def validate_step_id(step_id: str) -> str:
    """Validates a workflow step identifier. See :func:`validate_path_identifier`."""
    return validate_path_identifier(step_id, "step_id")
