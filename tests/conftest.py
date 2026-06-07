"""Shared pytest configuration for deterministic JAX test execution."""

from __future__ import annotations

import os


os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
