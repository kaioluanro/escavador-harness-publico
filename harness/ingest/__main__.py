"""Permite ``uv run python -m harness.ingest …`` (mesma rotina de ``harness ingest``)."""

from __future__ import annotations

from harness.ingest.segmentar import main

if __name__ == "__main__":
    raise SystemExit(main())
