from __future__ import annotations

import importlib.metadata

import mini_transformer as m


def test_version():
    assert importlib.metadata.version("mini_transformer") == m.__version__
