"""
Test utility module
"""
import pytest
import regm.utils as utils


@pytest.mark.parametrize("vec", [[1, 2, 3], (1, 2, 3)])
@pytest.mark.parametrize("size", [2, 3])
def test_check_size(vec, size):
    if len(vec) != size:
        with pytest.raises(AssertionError):
            utils.check_size(vec, size)
    else:
        utils.check_size(vec, size)


@pytest.mark.parametrize("vec", [1, [1, 2, 3], None])
@pytest.mark.parametrize("size", [3])
@pytest.mark.parametrize("default_value", [1.0])
def test_default_vec_factory(vec, size, default_value):
    vec = utils.default_vec_factory(vec, size, default_value)
    assert len(vec) == size
