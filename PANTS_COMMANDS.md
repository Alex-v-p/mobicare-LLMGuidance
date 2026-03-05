# Pants cheat sheet (WSL / Linux)

This repo uses Pants as a **dev/build tool**. Runtime/deployment should be done via Docker Compose.

## Where to run Pants
- Run inside Ubuntu/WSL **in the Linux filesystem** (e.g. `~/repos/...`), not under `/mnt/c/...`.

## Help / discovery
- `pants help`
- `pants help goals`
- `pants help <goal>` (e.g. `pants help test`)
- `pants --version`

## Core workflow
- Format: `pants fmt ::`
- Lint: `pants lint ::`
- Typecheck (if enabled): `pants check ::`
- Tests: `pants test ::`
- Run a target: `pants run path/to:target`
- Package/build artifact: `pants package ::`

## Useful inspection commands
- List targets: `pants list ::`
- Show a target’s resolved config: `pants peek path/to:target`
- What depends on what: `pants dependencies path/to:target`
- What files a target owns: `pants filedeps path/to:target`
- Create BUILD files automatically (starter): `pants tailor ::`

## Dependency / requirements (Python)
### If you use lockfiles (recommended for reproducibility)
1) Define resolves + lockfiles in `pants.toml`
2) Generate: `pants generate-lockfiles`
3) Update: `pants generate-lockfiles --resolve=<name>`

### If you’re just getting started
- Put dependencies in a `requirements.txt`
- Use `python_requirements()` in the nearest `BUILD` file
- Add those requirement targets as deps of your binaries/tests as needed

## Cache / cleanup
- Clean cached artifacts (safe): `pants clean-all`

## Common target patterns
- Run everything in repo: `pants test ::`
- Run only a folder: `pants test services/api::`
- Run one test file (example): `pants test services/api/src/api/tests/test_x.py`
- Run one binary: `pants run src/hello:hello`
