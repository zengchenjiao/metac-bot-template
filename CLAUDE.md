# Project Conventions

## Directory Structure

| Type | Directory |
|------|-----------|
| Core forecasting modules | `forecaster/` |
| Configuration | `config/` |
| Training data & optimization scripts | `training/` |
| Utility scripts | `tools/` |
| Markdown documents | `md/` |
| Training data JSON / optimized models | `json/` |
| GitHub Actions workflows | `.github/workflows/` |

## Rules

- New markdown files (`.md`) go into `md/`
- New training-related Python files go into `training/`
- New forecasting modules go into `forecaster/`
- New utility/tool scripts go into `tools/`
- Configuration changes go into `config/settings.py`
- `main.py` stays at project root (referenced by GitHub Actions)
