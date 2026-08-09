# Repository Guidelines

## Project Structure & Module Organization

Application code uses a `src` layout under `src/paper_rag/`. Shared configuration lives in `src/paper_rag/config.py`; focused helpers belong in `src/paper_rag/utils/`. Mirror new modules with tests in `tests/` (for example, `src/paper_rag/utils/ids.py` is covered by `tests/test_ids.py`). Runtime defaults are defined in `config/default.yaml`. Treat `COURSE_PLAN.md`, `LEARNING_STATE.md`, and `SOURCE_MANIFEST.md` as the rebuild roadmap, current checkpoint, and source inventory. Longer implementation plans belong in `docs/superpowers/plans/`.

## Build, Test, and Development Commands

- `uv sync --extra dev`: create or update the local environment with test and lint tools.
- `uv run pytest -q`: run the full test suite with concise output.
- `uv run pytest tests/test_config.py -q`: run one focused test module while iterating.
- `uv run ruff check .`: check imports, style, and common Python errors.
- `uv run ruff format --check .`: verify formatting without changing files; use `uv run ruff format .` to apply it.
- `uv build`: build source and wheel distributions using the configured setuptools backend.

## Coding Style & Naming Conventions

Target Python 3.10+ and use four-space indentation. Ruff enforces a 100-character line length, import ordering, pyupgrade, bugbear, simplify, and Ruff-specific rules. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and leading underscores for private helpers. Keep modules focused, add type hints to public interfaces, and prefer `pathlib.Path` for filesystem work. Comments and docstrings may be Chinese or English, but should remain concise and consistent within a module.

## Testing Guidelines

Pytest discovers `tests/test_*.py` and functions named `test_*`; strict markers are enabled. Add behavior-focused tests for every change, using fixtures such as `tmp_path` and `monkeypatch` to isolate files and environment variables. Run the focused test first, then the complete suite and Ruff checks. No numeric coverage threshold is currently configured, so prioritize meaningful branch and failure-path coverage.

## Commit & Pull Request Guidelines

History follows Conventional Commit subjects such as `feat(config): ...`, `fix(utils): ...`, and `docs(course): ...`. Keep commits scoped to one coherent change and do not overwrite unrelated working-tree edits. Pull requests should state the purpose, summarize key behavior or configuration changes, list verification commands, and link relevant issues. Include screenshots only when a user-visible interface changes.

## Security & Configuration

Never commit `.env` files, credentials, PDFs, model weights, databases, or runtime `data/` artifacts. Add shareable defaults to `config/default.yaml`; inject secrets through environment placeholders.
