"""Ingestion sources for NOESIS memory."""

from .web import ingest_web
from .local import ingest_local

__all__ = ["ingest_web", "ingest_local"]
