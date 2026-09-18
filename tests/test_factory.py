import pytest

from edgevision.factory import build_detector


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unknown backend"):
        build_detector({"backend": "openvino", "paths": {"openvino": "x"}})
