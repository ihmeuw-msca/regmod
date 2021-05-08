"""
Test NamedList
"""
import pytest

from regmod.composite_models.collections import NamedList


# pylint:disable=redefined-outer-name


@pytest.fixture
def nlist():
    return NamedList(["a", "b", "c"], [1, 2, 3])


def test_keys(nlist):
    assert nlist.keys == ["a", "b", "c"]


def test_values(nlist):
    assert nlist.values == [1, 2, 3]


@pytest.mark.parametrize("a", [(1, 4), ("1", 4)])
def test_append(nlist, a):
    nlist.append(*a)
    assert nlist.keys[-1] == "1"
    assert nlist.values[-1] == 4


def test_extend(nlist):
    nlist.extend(["d", "f"], [4, 5])
    assert nlist.keys[-2:] == ["d", "f"]
    assert nlist.values[-2:] == [4, 5]


@pytest.mark.parametrize("key", [-1, 2, "c"])
def test_pop(nlist, key):
    value = nlist.pop(key)
    assert value == 3
    assert len(nlist.keys) == 2
    assert len(nlist.values) == 2


@pytest.mark.parametrize(("key", "result"), [("a", True), ("d", False)])
def test_contains(nlist, key, result):
    assert (key in nlist) == result


@pytest.mark.parametrize("key", [-1, 2, "c"])
def test_getitem(nlist, key):
    assert nlist[key] == 3


@pytest.mark.parametrize("key", [-1, 2, "c"])
def test_setitem(nlist, key):
    nlist[key] = 4
    assert nlist[key] == 4


def test_len(nlist):
    assert len(nlist) == 3


def test_iter(nlist):
    assert nlist.values == list(i for i in nlist)
