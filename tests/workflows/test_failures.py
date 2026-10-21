# mypy: ignore-errors
import pickle

import pytest

from modal_workflows.workflows.failures import failure_from_exception, is_workflow_item_failure


class _TwoArgError(Exception):
    """Takes two args but passes one combined message to ``super().__init__()``."""

    def __init__(self, code, detail):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class TestFailureFromException:
    def test_multi_arg_exception_survives_pickle_round_trip_as_failure_snapshot(self):
        original = _TwoArgError("E42", "disk full")

        # Pickling the raw exception succeeds, but unpickling calls cls(*exc.args) with
        # too few args and raises, which is why failures are stored as dict snapshots.
        with pytest.raises(TypeError):
            pickle.loads(pickle.dumps(original))

        failure = failure_from_exception(original)
        roundtripped = pickle.loads(pickle.dumps(failure))

        assert is_workflow_item_failure(roundtripped)
        assert roundtripped["exc_type"].endswith("_TwoArgError")
        assert roundtripped["message"] == "E42: disk full"
