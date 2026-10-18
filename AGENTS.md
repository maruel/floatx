# floatx

Run `make help` for the build and check targets. Run `make fix`, `make verify`,
and `make test` before committing. CI also runs the platform and compatibility
checks in `.github/workflows/test.yml`. Install versioned hooks with
`make git-hooks`.

`make test` runs coverage and benchmark smoke checks and writes ignored
`coverage.txt`. Tests use a 600-second timeout.

CI shares checkout, locked Python execution and Go 1.27 across one Ubuntu check
job.

Go 1.27.0 or newer is required. golangci-lint v2.14.0 is tracked in go.mod and
runs with `go tool golangci-lint`.

The shared script index is generated with
`uv run --locked python scripts/update_agents_file_index.py`; use `--check` to validate it.

Python scripts and Ruff run through `uv run --locked`, which automatically syncs
the locked environment. `pyproject.toml` and `uv.lock` define the Python tools,
and `.editorconfig` defines shared editor defaults.
