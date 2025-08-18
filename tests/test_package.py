from __future__ import annotations

import importlib.metadata

import mini transformer as m


def test_version():
    assert importlib.metadata.version("mini transformer") == m.__version__
