"""Decorator registry. Adding an implementation = one file + one decorator.

Registries are declared up front even though most are empty at S1; each later
stage fills exactly one of them, which is the mechanical form of "one Protocol
per stage".
"""
from __future__ import annotations

from collections.abc import Callable


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'. available: {sorted(self._items)}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


LLMS: Registry = Registry("llm provider")          # S6
SOURCES: Registry = Registry("source adapter")     # S2
CHUNKERS: Registry = Registry("chunker")           # S3
TRANSFORMS: Registry = Registry("pack transform")  # S3
RETRIEVERS: Registry = Registry("retriever")       # S5
EMBEDDERS: Registry = Registry("embedder")         # S5b (dense / hybrid)
AGENTS: Registry = Registry("agent mode")          # S6
SCORERS: Registry = Registry("scorer")             # S8