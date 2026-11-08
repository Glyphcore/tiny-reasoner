import os, sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TR_GPU"):
        return
    skip = pytest.mark.skip(reason="set TR_GPU=1 to run gpu tests")
    for it in items:
        if "gpu" in it.keywords:
            it.add_marker(skip)
