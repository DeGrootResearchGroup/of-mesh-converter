# of-mesh-converter — Developer Guide

> **Doc maintenance:** when finishing any task that changes the build
> layout, public-facing names (CLI flags, dictionary keys), supported
> CGNS element types, supported output paths, the tutorial set, or
> the build/CI workflow, update **both** `CLAUDE.md` and `README.md`
> in the same change. The two files overlap intentionally — this
> guide is the long form, the README is the entry point — and they
> drift out of sync quickly if only one is touched. Quick check
> before committing: `grep` for any name or path you renamed in the
> other file.

> **Tests with features:** every new feature must ship with a test.
> Bug fixes that change behaviour need a regression test that would
> have failed before the fix. The `tests/` tree is what CI runs on
> every PR; the `tutorials/` tree is for pedagogical walkthroughs
> and is **not** run by CI. If a tutorial demonstrates a code path
> CI doesn't already cover, add a small bit-for-bit test under
> `tests/` so coverage of that path is preserved.

> **Code comments:** don't leave behind comments that only make sense
> if the reader saw the previous version. Notes like "# substr's
> second arg is length, not end index" right above corrected code,
> or "# fixed sign error" above a now-correct formula, read as
> nonsense to anyone arriving fresh — the broken code they reference
> is gone. The "why" of a fix belongs in the commit message, not the
> source. In-code comments should explain non-obvious invariants that
> hold *now*, not the bug that motivated the change.

## Project Overview

A standalone CLI tool that converts a finished ANSYS Fluent flow
solution (exported to CGNS) into the minimum set of OpenFOAM case
artefacts needed to drive the `radiationDose` Lagrangian tracker from
the sibling [photoBio](https://github.com/DeGrootResearchGroup/photoBio)
project.

The tool is deliberately scoped narrow: it is **not** a general-purpose
Fluent → OpenFOAM converter and should never grow into one. Every BC
type, turbulence model, and field name we map is one we can justify by
"the dose tracker needs it." The goal is one specific adoption story
(see "Strategic context" below), not a vendor-bridging product.

This project has **no compile-time or runtime dependency on the
photoBio repository.** The output it produces is a normal OpenFOAM
case directory that any OF13 install can read; we validate against
photoBio's `radiationDose` because that's the driver case we care
about, not because the tools are coupled.

## Strategic context

The friction for a Fluent shop adopting `radiationDose` isn't the dose
math — it's that switching from Fluent to OpenFOAM means re-validating
the *flow* solution that's already signed off, audited, and matched
against tracer studies. If we can let users keep their validated
Fluent flow and bolt on a better dose tracker, the adoption barrier
collapses from "rebuild everything" to "try this on a case I've
already run."

This converter is the artefact that makes that pitch concrete. The
explicit consequence: `radiationDose` becomes usable as a
post-processor on top of any commercial CFD result, which is
strictly broader than its current "drop-in for OpenFOAM users" reach.

Scope discipline matters here. A general "Fluent → OpenFOAM"
converter is a tarpit that's eaten projects before — every BC type
mapping is a year of edge cases. We dodge that by refusing to grow
features that aren't required by the dose tracker's input contract.

## Inputs and outputs

**In:** a single CGNS file containing
- unstructured mesh with zone (patch) family names preserved
- cell-centred `U` (m/s, vector)
- cell-centred `k` (m²/s²) and `ε` (m²/s³) — if the case ran a
  k-ε family turbulence model
- optionally cell-centred `G` (W/m²) — if the user already has a
  Fluent DO radiation solve they want to feed forward

**Out:** a ready-to-run OpenFOAM case skeleton
- `constant/polyMesh/{points,faces,owner,neighbour,boundary}`
- `0/U`, `0/k`, `0/epsilon`, optionally `0/G`
- `system/{controlDict,fvSchemes,fvSolution}` stubs sufficient for
  `foamPostProcess` to run
- `system/postProcess.dict` with a `radiationDose` block templated
  to the patch names just imported, leaving seeding / escape /
  `kInact` for the user to fill in

The output is a *skeleton*, not a copy of a tutorial. The user
supplies the dose-side config (seeding model, escape patches,
inactivation rate constants). The skeleton's only job is to ensure
the patch names referenced in the templated dose dict exist in the
imported `boundary` file.

## Architecture

**Python tool**, not a C++ OpenFOAM utility. Three reasons, in order
of weight:

1. CGNS parsing in C++ is a fight (the CGNS Mid-Level Library has a
   C API, the obvious wrappers are dormant — see "Prior art" below).
   In Python it's `pycgns` + `h5py`.
2. Keeping CGNS / HDF5 out of the photoBio C++ library build means
   users who haven't built `libopticalRadiation` or `libradiationDose`
   yet can still try this converter on a Fluent result — which is the
   whole adoption pitch. The OpenFOAM mesh format is plain text, so
   the writer needs no OF runtime linkage.
3. The conversion is one-shot, not in a tight loop. The Python ↔
   C++ performance gap doesn't bite.

**Layout (proposed):**

```
of-mesh-converter/
  CLAUDE.md
  README.md                       # user-facing entry point
  pyproject.toml                  # PEP 621, deps: pycgns, h5py, numpy
  src/of_mesh_converter/
    __init__.py
    __main__.py                   # CLI entry: `python -m of_mesh_converter`
    cgns_reader.py                # CGNS → in-memory mesh + fields
    foam_writer.py                # in-memory → polyMesh + 0/* fields
    elements.py                   # CGNS element-type connectivity tables
    field_mapping.py              # CGNS quantity-name → OF field-name dict
    sanitise.py                   # patch-name fixups, k/eps clipping
    sanity_report.py              # printed audit at end of run
  tests/
    test_elements.py              # connectivity tables vs CGNS spec
    test_field_mapping.py         # name mapping round-trip
    test_self_roundtrip.py        # photoBio doseSmokeBox → CGNS → back
    test_synthetic_fluent.py      # hand-built CGNS matching Fluent layout
    fixtures/
      doseSmokeBox.cgns           # generated, gitignored if large
```

The split between `cgns_reader.py` and `foam_writer.py` is hard: the
reader produces a format-neutral intermediate representation
(numpy arrays + a small set of dataclasses), the writer consumes it
and knows nothing about CGNS. Lets us test each side in isolation
and lets us add other readers (CGNS from STAR-CCM+, e.g.) later
without touching the writer.

The conversion pipeline:

1. **Read mesh.** Parse CGNS `Zone_t` unstructured zone. Handle
   `NGON_n` / `NFACE_n` (polyhedral) plus standard element types
   `TETRA_4`, `HEXA_8`, `PENTA_6`, `PYRA_5`. Polyhedral is the
   load-bearing path — most Fluent production meshes are poly-hex
   and the existing converters can't read them (see "Prior art").
2. **Read fields.** Parse `FlowSolution_t` cell-data arrays. Apply
   the dictionary-driven name mapping (CGNS uses
   `VelocityX/VelocityY/VelocityZ` and `TurbulentEnergyKinetic`,
   OpenFOAM uses `U` and `k`).
3. **Sanitise.** Clip `k <= 0` and `ε <= 0` to `1e-12` (Fluent
   occasionally emits these near walls; DRW produces NaN otherwise).
   Sanitise CGNS zone family names into valid OF patch identifiers
   (`interior-fluid` → `interior_fluid`) and print the mapping.
4. **Write polyMesh.** Build OF's face-based representation
   (`points`, `faces`, `owner`, `neighbour`, `boundary`) from the
   CGNS connectivity. Owner < neighbour ordering on internal faces.
5. **Write fields.** Emit `0/U`, `0/k`, `0/epsilon` as
   `volScalarField` / `volVectorField` with `internalField nonuniform
   List<...>` and `boundaryField` of `zeroGradient` everywhere. The
   fields are frozen inputs to a Lagrangian tracker; we never solve
   an equation on them, so BC type fidelity does not matter.
6. **Write system stubs.** Minimal `controlDict`, `fvSchemes`,
   `fvSolution` plus a templated `postProcess.dict` with the
   imported patch names plugged in.
7. **Sanity report.** Print mesh bounding box, cell count, per-patch
   face counts, per-field min/max, count of cells where `k` or `ε`
   needed clipping. This is the artefact the user shows their QA
   lead.

## Prior art

Researched 2026-05-15 before scoping this project. Findings are
load-bearing for the "fresh Python, don't port existing C++" call.

- **OpenFOAM v13 Foundation ships no CGNS converter.** The available
  mesh converters are `fluentMeshToFoam`, `starToFoam`,
  `gambitToFoam`, `ideasToFoam`, `cfx4ToFoam`. `fluentMeshToFoam` and
  `fluent3DMeshToFoam` are documented as flaky on polyhedral output;
  the engineer driving this project's requirements specifically
  called this out.
- **[SiHubb/cgnsToFromFoam](https://github.com/SiHubb/cgnsToFromFoam) and
  [wyldckat/cgnsToFromFoam](https://github.com/wyldckat/cgnsToFromFoam)
  exist but are dead ends for our case.** Both are ports of the
  original Hydro-Québec / TurboMachinery SIG `cgnsToFoam` from ~2008.
  Last commit on both: July 2018. Target OpenFOAM 5.x (we're on
  v13). Depend on a vendored `libcgnsoo` C++ wrapper from the CVS
  era that would itself need porting before any of the conversion
  logic ran. **Critically**, the converter does not support `NGON_n`
  / `NFACE_n` polyhedral elements — see
  [`CGNSElementType.H`](https://github.com/SiHubb/cgnsToFromFoam/blob/OF5x/applications/utilities/mesh/conversion/cgnsToFoam/CGNSElementType.H),
  which returns -1 for `NGON_n` and never maps it to an OF cell
  model. The very element type Fluent users need (poly-hex) is the
  one this codebase can't read. Resurrecting it would cost ~2-3
  weeks of work against an unfamiliar 2018 C++ codebase before
  adding any value over what's there.
- **[meshio](https://github.com/nschloe/meshio) reads CGNS and writes
  OpenFOAM polyMesh, but the polyMesh writer is incomplete** and
  polyhedral support is weak in both directions ([issue #660](https://github.com/nschloe/meshio/issues/660)).
  Same blocker from a different direction.
- **Two concepts worth lifting from `cgnsToFromFoam`** (and
  reimplementing fresh, not copying):
  - The dictionary-driven CGNS-quantity-name → OF-field-name mapping
    (`CgnsToFoamDictionary.H` in the original). The exact mapping
    we need (`VelocityX/Y/Z` → `U`, `TurbulentEnergyKinetic` → `k`,
    `TurbulentDissipationRate` → `epsilon`) is in the CGNS Standard
    Interface Data Structures (SIDS) document, but the design
    pattern of pushing it into a user-overridable dict is the right
    one and worth preserving.
  - The split into separate modules for mesh element handling, field
    conversion, and connectivity mapping. Confirms the architecture
    above. We are not copying the code, only the structural choice.
- **Standard-element connectivity tables (CGNS node ordering for
  TETRA_4, HEXA_8, etc.) are CGNS-spec tables** and don't go stale.
  Pull them from the CGNS SIDS document, cross-check against
  `pycgns` constants, and unit-test against a known mesh.

## Scope

### In scope (v1)

- CGNS input only (single file, single `Zone_t` unstructured grid).
- Steady flows only (one `FlowSolution_t`).
- Realisable / standard / RNG k-ε turbulence — anything that exports
  `k` and `ε` directly.
- Polyhedral, hex, tet, prism, pyramid cells.
- Patch name preservation (with documented sanitisation).
- One self-roundtrip CI test and one synthetic-Fluent-layout test.

### Explicitly out of scope

- Direct `.cas` / `.dat` binary parsing. CGNS only. Users export
  from Fluent once.
- SST k-ω → k-ε conversion. v1 errors out with a clear message
  pointing the user at switching their Fluent solver. Revisit if a
  driver case appears.
- Transient flows.
- BC type fidelity for the imported fields. They're frozen inputs to
  a Lagrangian tracker; `zeroGradient` boundaries are correct because
  `radiationDose` never solves an equation on them.
- STAR-CCM+, OpenFOAM-ESI variants, anything not Fluent + CGNS. The
  reader/writer split makes adding other readers tractable later;
  doing it pre-emptively is feature-creep.
- General-purpose Fluent → OF conversion. Every BC type mapping
  beyond what the dose tracker needs is a year of edge cases.

## Validation

Three tiers, in order of fidelity. CI runs only tier 1; tiers 2 and
3 are partner-dependent.

### Tier 1 — self-roundtrip (CI)

Take `tests/doseSmokeBox` from the photoBio repo (1 m × 0.1 m × 0.1 m
slip-wall box with uniform `U = (0.5, 0, 0)`, deterministic 2.0
mJ/cm² dose for every particle — see the photoBio CLAUDE.md). Export
it to CGNS via either `foamToCGNS` (if available on the user's
install) or a hand-written conversion script. Run `of-mesh-converter`
on the CGNS file. Assert:

- mesh `points` / `faces` / `owner` / `neighbour` reproduce to
  bit-equality
- `0/U` cell-centred values reproduce to floating-point
- patch names round-trip with the documented sanitisation rules
- running `radiationDose` on the converted case reproduces the
  2.0 mJ/cm² answer to 1 part in 1000 (same tolerance as
  `tests/doseSmokeBox/validate`)

Cheap, no Fluent licence needed, lands as a CI test on every PR.
Catches regressions in the reader / writer layout without external
dependencies.

### Tier 2 — synthetic Fluent CGNS

Hand-build a small CGNS file matching the layout Fluent's exporter
produces — zone family naming, NGON_n/NFACE_n ordering, dataset
hierarchy, units. The reference for the layout is real Fluent
exports collected from users plus the CGNS SIDS. Plug numbers in
that match a plug-flow analytical case and assert the dose tracker
reproduces the analytical answer.

This is the test that proves we can read what Fluent actually emits,
distinct from what the CGNS spec allows in theory. Build it from
real exports the engineer can provide.

### Tier 3 — Sozzi-in-Fluent

Partner with a Fluent user (the engineer driving this scoping
exercise is a candidate) to solve `tutorials/uvReactorSozzi2006`'s
geometry in Fluent, export to CGNS, run `of-mesh-converter` +
`radiationDose`, document the deviation vs the photoBio-native
answer (mean dose 70.28 mJ/cm² analytical, 64.45 mJ/cm² DOM-driven;
log reduction at `kInact = 0.1 cm²/mJ` = 2.05 / 1.39 respectively).

This is the artefact for QA leads — the answer to "I have a Fluent
result, does this thing give me a sensible dose?" Out-of-band from
CI; depends on Fluent licence access and partner availability.

## Risks

- **CGNS polyhedral storage (`NGON_n` / `NFACE_n`) is less mature in
  tooling than tet/hex.** Fluent emits it but the indexing
  conventions deserve a careful read against the CGNS SIDS before
  any code lands. Cross-check against at least one real Fluent
  export before declaring the reader done.
- **Patch-name collisions.** Fluent zone names like `interior-fluid`
  contain characters OF won't accept. We need a documented
  sanitisation rule (e.g. `[^a-zA-Z0-9_]` → `_`, leading-digit
  prefixed with `p_`) and a printed mapping report so the user can
  reconcile their dose-dict patch names.
- **Near-wall negative `ε`.** Already noted under sanitise step.
  Default action: clip to `1e-12`, print a warning with cell count.
- **CGNS exports vary by Fluent version.** v1 targets Fluent 2024
  R1+; older versions get a "may work, untested" disclaimer in the
  README. Don't promise what we can't validate.
- **Fluent licence dependency for tier 2 / 3 validation.** The
  in-house team doesn't have one. Mitigation: build tier 1 on the
  bit-for-bit photoBio roundtrip so CI carries the load; lean on
  external partners for the real-Fluent tiers.

## Effort estimate

- Mesh conversion (CGNS unstructured → polyMesh) including
  polyhedra: ~3-5 days. Dominated by getting NGON_n / NFACE_n face
  ordering right. Standard-element connectivity is a day at most.
- Field conversion: ~1 day. Dictionary-driven, mechanical.
- Sanity report + patch sanitisation: ~0.5 day.
- Self-roundtrip validation harness wired into CI: ~1 day.
- README + usage examples: ~0.5 day.

**Total in-house: ~1-1.5 weeks** to a v1 that passes the
self-roundtrip and ingests a synthetic Fluent CGNS file. The
Sozzi-in-Fluent tier is the bottleneck on shipping with confidence
to external users, and is partner-dependent.

## Development workflow

Code review goes through GitHub PRs against `main`. The Claude Code
sandbox doesn't have an SSH key for `git@github.com:DeGrootResearchGroup/...`
and `gh` isn't installed, so Claude can't push branches or open PRs
directly. The convention (lifted from the sibling photoBio repo) is:

1. Claude commits changes on a sensibly-named local branch (e.g.
   `add-<feature>`, `fix-<thing>`, `<area>-<change>`) — never on
   `main`, never on the `claude/<worktree-name>` scratch branch.
2. The user pushes that branch from their own checkout and opens
   the PR. From a Claude worktree the branch can be picked up with
   `git fetch <worktree-path> <branch>:<branch>` and then
   `git push -u origin <branch>` from the main checkout, or by
   `cd`-ing into the worktree and pushing if the user's shell has
   the SSH key.

Don't try to `git push` from inside the sandbox; it fails with
"Permission denied (publickey)" and wastes a turn.

### CI

`.github/workflows/ci.yml` runs `pytest -ra` on every PR and push to
`main`, across Python 3.10 / 3.11 / 3.12 on `ubuntu-latest`. The
suite is pure-Python — no Docker, no OpenFOAM install, no Fluent
licence — so it runs in well under a minute. The two heavy
dependencies (`numpy`, `h5py`) install cleanly from PyPI wheels.

### Tutorials vs tests

The case suite is split into two trees:

- **`tests/`** — regression suite, run by CI on every PR. Synthetic
  CGNS files generated by `tests/cgns_fixture.py` plus unit tests on
  the individual modules. This is what you re-run when fixing a bug
  and what gates merges.
- **`tutorials/`** — pedagogical / walkthrough cases, run on demand
  by the user via `python <tutorial>/run.py` (or whatever the
  tutorial's entry-point is). Not run by CI. Each tutorial has a
  `README.md` describing what it demonstrates. If a tutorial covers
  a code path CI doesn't already exercise, add a small replacement
  test under `tests/` so coverage of that path is preserved when
  the tutorial bitrots.

New features must always be shipped with tests in `tests/`. Bug
fixes that change observable behaviour need a regression test that
would have failed before the fix.

## Relationship to photoBio

Sibling project. No code dependency in either direction. The link
is:

- This tool's tier 1 validation reads a test case from the photoBio
  repo (`tests/doseSmokeBox`) and rounds it through CGNS. If
  photoBio's `doseSmokeBox` changes shape, the validation case needs
  re-exporting — but the converter itself doesn't break.
- Users who want to chain Fluent → `of-mesh-converter` →
  `radiationDose` need both repos installed. The README should
  document this end-to-end pipeline as the main use case.
- If photoBio's `radiationDose` input contract changes (e.g. a new
  required field), the `postProcess.dict` template emitted by
  step 6 needs updating. Catch this in CI by depending on the
  photoBio repo for the validation case, not by trying to mirror
  its dictionary schema.

## Open questions

These are decisions to make during the build session, not to
pre-commit to here. Listed so a fresh session knows what to think
about before writing code.

1. **Mesh IR format.** Plain dataclasses + numpy arrays, or a
   first-class `Mesh` class with methods? The simpler path is
   probably fine for a one-shot converter.
2. **`pycgns` vs raw `h5py`.** `pycgns` is the official Python
   binding but installs awkwardly on some platforms; `h5py` is
   ubiquitous and CGNS is just an HDF5 file with a documented
   schema. Trying both early would lower risk.
3. **OF case-skeleton authoring strategy.** Hand-write the
   `controlDict` / `fvSchemes` / `fvSolution` stubs as Python
   string templates, or copy from a known-good reference case in
   the repo? String templates are simpler and have no external
   dependency.
4. **Should the tool be pip-installable, or just runnable from a
   checkout?** Pip-installable is the cleaner distribution story
   for external Fluent users (the strategic use case) but adds
   packaging cost. Start with checkout-runnable and add packaging
   when the tier 3 validation lands.
5. **Git repo: initialise now or after first commit?** Out of scope
   for this doc; whoever starts the build session decides.

## Reference material

- [CGNS Standard Interface Data Structures (SIDS)](https://cgns.github.io/CGNS_docs_current/sids/index.html)
  — authoritative spec for element types, zone layouts, conventions.
- [pyCGNS documentation](http://pycgns.github.io/)
- [OpenFOAM v13 mesh description](https://doc.cfd.direct/openfoam/user-guide-v13/mesh-description-1)
  — owner / neighbour / boundary file formats.
- [SiHubb/cgnsToFromFoam](https://github.com/SiHubb/cgnsToFromFoam) —
  reference only, do not port. Useful for the dictionary-driven
  field-mapping pattern and the standard-element connectivity
  tables. Last touched July 2018.
- [photoBio repository](https://github.com/DeGrootResearchGroup/photoBio)
  — sibling project hosting `radiationDose`, the driver use case.
  See its `CLAUDE.md` for the dose tracker's input contract.
