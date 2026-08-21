# mypy: ignore-errors
import pytest

from modal_workflows.workflows.identifiers import (
    MAX_IDENTIFIER_LENGTH,
    validate_path_identifier,
    validate_state_name,
    validate_step_id,
)


class TestValidatePathIdentifier:
    @pytest.mark.parametrize(
        "value",
        [
            "train",
            "step_1",
            "test-run",
            "ap-abc123_fc-def456",
            "map_double_12",
            "checkpoint.v2",
            "a",
            "0",
            "x" * MAX_IDENTIFIER_LENGTH,
        ],
    )
    def test_accepts_valid_identifiers(self, value):
        assert validate_path_identifier(value, "step_id") == value

    @pytest.mark.parametrize(
        "value",
        [
            "..",
            "../etc",
            "../../etc/passwd",
            "a/../../b",
            "nested/step",
            "/etc/passwd",
            "back\\slash",
            ".hidden",
            "with\x00null",
            "with space",
            "semi;colon",
            "tilde~",
            "dollar$",
            "new\nline",
            "unicode-é",
        ],
    )
    def test_rejects_traversal_and_separators(self, value):
        with pytest.raises(ValueError, match="Invalid step_id"):
            validate_path_identifier(value, "step_id")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_path_identifier("", "step_id")

    def test_rejects_too_long(self):
        with pytest.raises(ValueError, match="at most 128 characters"):
            validate_path_identifier("x" * (MAX_IDENTIFIER_LENGTH + 1), "step_id")

    @pytest.mark.parametrize("value", [None, 1, ("tuple",)])
    def test_rejects_non_strings(self, value):
        with pytest.raises(ValueError, match="must be a string"):
            validate_path_identifier(value, "step_id")

    def test_error_message_names_the_identifier_kind(self):
        with pytest.raises(ValueError, match="Invalid state_name"):
            validate_path_identifier("../escape", "state_name")


class TestNamedValidators:
    def test_validate_state_name_returns_value(self):
        assert validate_state_name("test-run") == "test-run"

    def test_validate_state_name_rejects_traversal(self):
        with pytest.raises(ValueError, match="Invalid state_name"):
            validate_state_name("../../mnt")

    def test_validate_step_id_returns_value(self):
        assert validate_step_id("train") == "train"

    def test_validate_step_id_rejects_traversal(self):
        with pytest.raises(ValueError, match="Invalid step_id"):
            validate_step_id("../../train")
