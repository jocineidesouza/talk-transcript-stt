# Repository Guidelines

## Project Structure & Module Organization
This repository is currently minimal and does not yet define an application layout. Keep new code organized from the start:

- `src/` for application or library code
- `tests/` for automated tests
- `docs/` for design notes and contributor-facing documentation
- `assets/` for static files such as images or sample data

Prefer small, focused modules. Match test files to source paths, for example `src/parser.js` with `tests/parser.test.js`.

## Build, Test, and Development Commands
No build system is configured yet. When adding one, expose a small, stable command set and document it here. Recommended defaults:

- `make test` or `npm test`: run the full test suite
- `make lint` or `npm run lint`: run formatting and lint checks
- `make build` or `npm run build`: produce a distributable build
- `make dev` or `npm run dev`: start a local development workflow

Contributors should avoid introducing ad hoc scripts when a standard project entry point will do.

## Coding Style & Naming Conventions
Use consistent formatting and keep style automated. Unless the stack requires otherwise:

- Indent with 2 spaces for YAML, JSON, and Markdown; use the language-standard convention elsewhere
- Use `snake_case` for file names in scripts and docs
- Use `PascalCase` for classes and `camelCase` for functions and variables
- Keep modules single-purpose and avoid large utility dumps

If you add a formatter or linter, commit its config with the same change.

## Testing Guidelines
Add tests alongside new functionality. Prefer fast, deterministic unit tests before broader integration coverage.

- Name tests after the unit under test, such as `tests/auth.test.js` or `tests/test_auth.py`
- Cover happy paths, edge cases, and failure behavior
- Do not merge features without at least one automated test path

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so use a simple, imperative commit style:

- `feat: add parser entry point`
- `fix: handle missing config file`
- `docs: add contributor guide`

Pull requests should include a short summary, testing notes, linked issues when applicable, and screenshots only for UI changes.

## Configuration & Security
Do not commit secrets, credentials, or local environment files. Add new configuration keys to a checked-in example file such as `.env.example` and document required values in `docs/`.
