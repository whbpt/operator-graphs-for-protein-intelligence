import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_seqmodels.py"
SPEC = importlib.util.spec_from_file_location("convert_seqmodels", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_validate_family_and_slugify() -> None:
    family = {
        "x_id": "PF/001 test",
        "x": np.asarray([[0, 1, 20], [1, 1, 2]], dtype=np.int64),
        "x_w": np.asarray([1.0, 0.5]),
        "x_true": np.eye(3),
        "x_mask": np.ones((3, 3), dtype=bool),
    }
    validated = MODULE.validate_family(family)
    assert validated["x"].dtype == np.uint8
    assert validated["x_w"].dtype == np.float32
    assert validated["x_mask"].dtype == bool
    assert MODULE.slugify(family["x_id"]) == "PF_001_test"


def test_validate_family_rejects_invalid_states() -> None:
    family = {
        "x_id": "bad",
        "x": np.asarray([[21]], dtype=np.int64),
        "x_w": np.asarray([1.0]),
        "x_true": np.zeros((1, 1)),
        "x_mask": np.ones((1, 1), dtype=bool),
    }
    with pytest.raises(ValueError, match="state range"):
        MODULE.validate_family(family)
