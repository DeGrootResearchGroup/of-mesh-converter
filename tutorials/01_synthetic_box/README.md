# Tutorial 01 — Synthetic hex box

Builds a small CGNS file in Python (no Fluent licence needed), runs
`of-mesh-converter` on it, and inspects the resulting OpenFOAM case.
This is the gentlest possible introduction to the tool.

## What gets built

A 1.0 × 0.1 × 0.1 m box meshed into 4×2×2 = 16 hex cells, with three
boundary patches:

- `inlet`  — the `x = 0` face, uniform `(0.5, 0, 0)` m/s velocity.
- `outlet` — the `x = Lx` face.
- `walls` — the remaining four faces (y-min, y-max, z-min, z-max).

Field state inside the box: uniform `U = (0.5, 0, 0)` m/s,
`k = 1e-3` m²/s², `ε = 1e-4` m²/s³ — a contrived but
dimensionally-consistent low-turbulence plug-flow snapshot. The
geometry and field set deliberately mirror `photoBio`'s
`tests/doseSmokeBox` so you can chain this tutorial into
`radiationDose` if you have OpenFOAM 13 + photoBio installed.

## Run it

```bash
cd tutorials/01_synthetic_box
python run.py
```

Output:

```
tutorials/01_synthetic_box/
├── case_out.cgns           # the synthetic CGNS file we built
└── case_out/               # the OpenFOAM case the converter produced
    ├── constant/polyMesh/{points,faces,owner,neighbour,boundary}
    ├── 0/{U,k,epsilon}
    ├── system/{controlDict,fvSchemes,fvSolution,postProcess.dict}
    └── conversion_report.txt
```

Both `case_out.cgns` and `case_out/` are `.gitignore`d.

## What to look at next

- **`case_out/conversion_report.txt`** — the audit the converter
  printed. Note the patch names, face counts, and per-field min/max.
  This is what you'd show a QA lead when reporting back on a real
  Fluent conversion.
- **`case_out/constant/polyMesh/boundary`** — confirm the three
  patches came through with the names you gave them.
- **`case_out/system/postProcess.dict`** — the `radiationDose`
  template. The patch names you'd plug into `escapePatches` /
  `seeding.patches` are already listed in the file's leading
  comment.

## Chain into radiationDose (optional)

If you have OpenFOAM 13 and the
[photoBio](https://github.com/DeGrootResearchGroup/photoBio)
library compiled, you can run the dose tracker on this output:

```bash
cd case_out
# Edit system/postProcess.dict: set seeding.patches to (inlet)
# and termination.escapePatches to (outlet), and pick a kInact value.
foamPostProcess -dict system/postProcess.dict -latestTime
```

The dose tracker writes per-particle CSVs and summary stats into
`postProcessing/radiationDose/<time>/`. Because this geometry's
`U`, `k`, `ε` are uniform and there's no `G` field, the tracker
will report `D = 0` for every particle — that's expected; this
tutorial is about the *converter*, not the dose physics.
