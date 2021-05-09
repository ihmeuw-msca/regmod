"""
Test NamedList
"""
import pytest

from regmod.composite_models.collections import NamedList


# pylint:disable=redefined-outer-name


@pytest.fixture
def nlist():
    return NamedList({"a": 1, "b": 2, "c": 3})


def test_default_init():
    nlist = NamedList()
    assert len(nlist) == 0


@pytest.mark.parametrize("key", [-1, 2, "c"])
def test_pop(nlist, key):
    value = nlist.pop(key)
    assert value == 3
    assert len(nlist.keys()) == 2
    assert len(nlist.values()) == 2


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
    assert list(nlist.values()) == list(i for i in nlist)
