# Tutorials

Pedagogical walkthroughs that show how to drive `of-mesh-converter`
end-to-end. **Not run by CI** — these are for users learning the
tool, not for regression coverage. The regression suite lives under
`../tests/` and is what every PR has to pass.

Each subdirectory is self-contained and produces a real OpenFOAM
case in its own `case_out/` directory when you run it. The `case_out`
trees are `.gitignore`d so the repo doesn't carry around generated
artefacts.

## Prerequisites

```bash
pip install -e ..[dev]    # from this repo's root
```

You **do not** need OpenFOAM installed to run the tutorials — they
produce OF-format files and stop there. To actually run
`radiationDose` on the output you also need the sibling
[of-optical-radiation](https://github.com/DeGrootResearchGroup/of-optical-radiation) repo and
an OpenFOAM 13 install.

## Walkthroughs

- **`01_synthetic_box/`** — Build a small CGNS file from scratch in
  Python, convert it with `of-mesh-converter`, and inspect the
  resulting OpenFOAM case. Useful for understanding what the
  converter actually produces without needing a Fluent licence.

Add a tutorial when you want a runnable example of a new feature.
If the tutorial exercises a code path the test suite doesn't, add a
matching test under `../tests/` so CI keeps a regression on that
path — tutorials drift; tests don't.
