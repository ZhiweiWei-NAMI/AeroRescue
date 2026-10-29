`SC01.json` is the representative configuration. Run from the repository root:

```
/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python tools/make_run.py SC01 --seed 1
python3 tools/summarize_run.py <run_dir>
```

`--materialize-only` writes AAS inputs and normalized display wrappers under
`tools/generated/` without starting AAS. `--config`, `--runtime`, and `--aas`
override paths. The launcher sets PYTHONPATH to `site:repo` for the CLI; there is
no installation into AAS's environment.

Shared `rescue.scene-config/v1` fields:

- `background`: id, canonical NPZ grid, build report, accepted scene list,
  accepted console source, delivery root, cited geometry source, named roads
  (`id`, `points` in EN metres, `width_m`) and rectangular named plots
  (`bounds_en` = `[Emin,Nmin,Emax,Nmax]`). Geometry annotations are data, not
  scene-specific Python. Non-rectangular terrain can still supply these
  candidate placement rectangles wholly within its valid semantic regions.
- `scenario_id`, `seed`, `tick_s`, `until_s`: catalog disaster and clock.
- `agents`: catalog embodiment kind, id, `start_en`, explicit `overrides`.
  Optional `task_allowlist` scopes generation only; optional public
  `navigation_goals` are deployment intentions evaluated by agent rules.
- `placement.buildings`: id, plot, anchor EN, jitter metres, size `[E,N]`, local
  damage grade D0–D3, occupant initial facts. Truth is hidden from agents.
- `placement.debris`: id, linked road/source building, fractional position range,
  allowed grid classes, volume, height, work, remaining road width.
- `placement.stations`: id, EN point, allowed grid classes, kind, capacity,
  power and explicit receiver confirmation policy.
- `display`: authored offsets above canonical terrain for UAVs, other agents,
  people and state labels. These offsets do not affect physical truth.
- `assumptions`: numerical and modelling assumptions carried into run results.

The adapter samples canonical heights bilinearly. It copies named background
roads/plots and checks authored placement against canonical classes; it does not
segment images or infer buildings from appearance. Other backgrounds need their
own cited geometry annotations and canonical class names in configuration.
Terrain slope/collision physics remain the v0.5 horizontal navigation model.

Inherited SC01 secondary events remain enabled, including the profile's scripted
420-second aftershock. They may reblock a road and prevent mission completion.
Actual passes and rescue stages are reconstructed from events; missing stages
are printed as missing, never filled from the intended chain.

A2 assets are read-only inputs. When present, the launcher creates display-only
GLB wrappers under its generated directory (grounded origin, forward heading,
configured building footprint), then stages those wrappers in the existing
runtime. Damage-specific building models and debris opening/cleared states have
separate registry types. Missing models use labelled markers with a recorded
reason. This lane has not run or validated the generated scenario.
