"""Importing the implementations here is what registers them."""
from . import bedrock, mock  # noqa: F401  (registration side effects)
from .base import Embedder, build_embedder

__all__ = ["Embedder", "build_embedder"]
