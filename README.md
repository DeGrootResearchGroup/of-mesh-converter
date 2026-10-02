# of-mesh-converter

Convert a finished ANSYS Fluent flow solution exported to CGNS into the
minimum OpenFOAM case skeleton needed to drive the `radiationDose`
Lagrangian tracker from the sibling
[of-optical-radiation](https://github.com/DeGrootResearchGroup/of-optical-radiation) project.

This is **not** a general-purpose Fluent → OpenFOAM converter. Every
boundary-condition type, turbulence model, and field name it knows
about is one the dose tracker actually consumes. The intent is to
collapse the adoption barrier for Fluent shops from "rebuild the flow
solution in OpenFOAM" to "try this on a CGNS export of a case you've
already validated."

## Status

v1, in active development. CI runs on synthetic CGNS, including a
polyhedral file in the layout Fluent writes. One real Fluent export
has been converted and checked: the Sozzi & Taghipour reactor on a
1.23M-cell polyhedral mesh (DO incident radiation only, no flow),
which `checkMesh` passes and whose cells match the OpenFOAM mesh it
was made from to within 1 nm. That export carried no `U`, `k` or
`epsilon`, so the flow-field path is still untested on real Fluent
output (see CLAUDE.md "Validation").

## Quick start

```bash
# Install (Python 3.10+).
pip install -e .

# Convert.
of-mesh-converter path/to/case.cgns path/to/case_dir/

# Or as a module.
python -m of_mesh_converter path/to/case.cgns path/to/case_dir/
```

The converter prints a sanity report to stdout and also writes it to
`case_dir/conversion_report.txt`. The report has the mesh bounding
box, cell count, per-patch face counts, per-field min/max/mean, any
patch-name renames, and a count of cells where `k` or `epsilon` had
to be clipped. **Show that report to your QA lead** — it's what
confirms the CGNS export actually rolled forward correctly.

After conversion you'll find:

```
case_dir/
├── constant/polyMesh/{points,faces,owner,neighbour,boundary}
├── 0/{U,k,epsilon[,G]}
├── system/{controlDict,fvSchemes,fvSolution,postProcess.dict}
└── conversion_report.txt
```

`system/postProcess.dict` is a `radiationDose` template with the
imported patch names plugged into the right slots. You still need to
fill in:

- Which patch is the inlet (the seeding model's `patches` list).
- Which patches are escape patches (typically the outlet).
- The `kInact` list for the organism(s) of interest.

Once those are filled in, run from `case_dir`:

```bash
foamPostProcess -dict system/postProcess.dict -latestTime
```

## Inputs the converter understands

- **Mesh**: a single CGNS file with one `CGNSBase_t` and one
  unstructured `Zone_t`, holding either
  - polyhedral cells: `NGON_n` faces plus `NFACE_n` cells, which is
    what Fluent writes for poly and poly-hex meshes. Both the CGNS
    4.x layout (`ElementStartOffset`) and the CGNS 3.x layout
    (length-prefixed entries) are read; or
  - linear standard elements: `TETRA_4`, `HEXA_8`, `PENTA_6`
    (wedge), `PYRA_5` (pyramid), with `TRI_3` / `QUAD_4` boundary
    sections.

  A zone mixing the two is rejected.
- **Boundary conditions**: each CGNS `BC_t` becomes one OpenFOAM
  patch. `BCWall*` types become OF `wall` patches; everything else
  becomes a generic `patch`. The dose tracker only reads patch
  *names* (for `escapePatches`, `seedingPatches`), so BC type
  fidelity beyond `wall` vs `patch` is not load-bearing. A BC's
  name is its `FamilyName` if it has one; Fluent writes none and
  names BCs `<zone>-<zone type>` (`lamp0_wall-wall`), so the
  converter takes the zone name from the BC's face section
  (`lamp0_wall-Pg`) instead. Fluent lower-cases zone names on
  export, which the converter cannot undo.
- **Mesh interfaces**: Fluent exports interface zones (non-conformal
  seams) as `BCWall`. They become uncoupled patches, and the report
  lists them: couple them in OpenFOAM before tracking on the
  converted mesh, or map the fields onto a coupled OpenFOAM case.
- **Fields**: cell-centred `VelocityX/Y/Z`, `TurbulentEnergyKinetic`,
  `TurbulentDissipationRate`, optional `Incident_Radiation`
  (Fluent's DO / P1 name), `FluenceRate` or `G`, all mapped to OF
  `U`, `k`, `epsilon`, `G`. Fluent sizes solution arrays by the
  zone's element count (faces plus cells) and indexes them by
  element id; the reader takes the cell slice and says so in the
  report. Arrays with one value per cell are read as they are.
  Anything else in the `FlowSolution_t` is ignored. Turbulence
  models other than k-ε (k-ω SST, RSM, ...) export different
  fields and are currently rejected — `radiationDose`'s
  discreteRandomWalk dispersion model needs `k` and `epsilon`
  directly.

## What gets sanitised

- **Patch names**. Anything outside `[A-Za-z0-9_]` becomes `_`,
  leading digits get a `p_` prefix, consecutive underscores
  collapse, and collisions get `_2`, `_3`, ... appended. The
  mapping is printed in the sanity report so you can reconcile
  your dose-dict `escapePatches` list against what actually got
  written.
- **Turbulence floors**. Fluent occasionally emits `k <= 0` or
  `epsilon <= 0` in near-wall cells. We clip both to `1e-12`. The
  count of clipped cells goes into the report; if it's >0 in
  cells away from walls, that's a flag to look at the Fluent
  solution.

## What's deliberately not in scope

- Direct `.cas` / `.dat` binary parsing. CGNS only; users export
  from Fluent once.
- SST k-ω → k-ε conversion. Switch the Fluent solver before
  exporting.
- Transient flows. `radiationDose` reads `U` and `G` once and
  holds them constant.
- BC type fidelity for the imported fields. Every patch gets
  `zeroGradient`. The fields are frozen inputs to a Lagrangian
  tracker; we never solve an equation on them.
- STAR-CCM+, OpenFOAM-ESI variants, anything not Fluent + CGNS.
  The reader/writer split makes adding other readers tractable
  later; doing it pre-emptively is feature-creep.

The full scope discussion (including why we *don't* try to be a
general-purpose Fluent→OF bridge) is in
[CLAUDE.md](CLAUDE.md#scope).

## Development

```bash
git clone <this repo>
cd of-mesh-converter
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

- **`tests/`** — regression suite run by GitHub CI on every PR via
  `pytest`. Unit tests for elements / field mapping / sanitise,
  plus integration tests that round-trip synthetic CGNS files
  through the full pipeline.
- **`tutorials/`** — pedagogical walkthroughs that show how to
  drive the converter from Python or the CLI. Not run by CI; meant
  for users learning the tool.

New features must always be shipped with tests; see CLAUDE.md
"Development workflow".

## Relationship to of-optical-radiation

Sibling project, no code dependency in either direction. The chain
is:

```
Fluent .cas/.dat
   ↓ (export)
case.cgns
   ↓ (of-mesh-converter)
OpenFOAM case directory
   ↓ (foamPostProcess -dict system/postProcess.dict)
radiationDose results
```

Users who want the full pipeline need both this repo and
[of-optical-radiation](https://github.com/DeGrootResearchGroup/of-optical-radiation)
installed. The dose-tracker side of the chain is documented in
of-optical-radiation's README and CLAUDE.md.

## License

MIT.
