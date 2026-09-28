"""Importing the implementations here is what registers them.

The decorator @SOURCES.register(...) only runs when its module is imported, so
an implementation that is never imported does not exist as far as the registry
is concerned. Adding an adapter = one file + one decorator + one line here.
"""
from . import domain_agent, jira, markdown_docs, pdf_docs  # noqa: F401
from .base import SourceAdapter, load_documents

__all__ = ["SourceAdapter", "load_documents"]