"""YAML config with inheritance (`base:`) and `key.sub=value` CLI overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import yaml


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def _read(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = raw.pop("base", None)
    if base:
        return _merge(_read(path.parent / base), raw)
    return raw


def _to_ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in d.items()})
    return d


def to_dict(ns) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: to_dict(v) for k, v in vars(ns).items()}
    return ns


def load_config(path: str, overrides: list[str] | None = None) -> SimpleNamespace:
    d = _read(Path(path))
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = d
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return _to_ns(d)
