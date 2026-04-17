#!/usr/bin/env -S just --justfile

@setup:
  @echo "Checking environment"
  uv lock --check
  uv pip install -e .

@test:
  @echo "Running pytest"
  uv run pytest -v

@push:
  #!/bin/bash
  PKG_VERSION=$(grep -E '^version\s*=' pyproject.toml | cut -d '"' -f2)
  git commit -m "v${PKG_VERSION}"
  git push origin main
  # Delete if failure
  # git tag -d v0.3.3
  # git push --delete origin v0.3.3
  git tag v${PKG_VERSION}
  git push origin v${PKG_VERSION}
