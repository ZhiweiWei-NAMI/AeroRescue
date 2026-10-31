# Asset baking and audit report

Generated: 2026-10-31T00:40:51.938674+00:00

Baked **359** categorized assets from **359** workbook-delivered geometry IDs. Recorded **0** unresolved source/selection/baking gaps. No gap is counted as a successful delivery.

| Category | Baked | Raw bytes | Baked bytes | Ratio | Draws before → after | Texture bytes before → after | Quantized |
|---|---:|---:|---:|---:|---:|---:|---:|
| AGT | 15 | 74,069,276 | 27,040,964 | 2.74× | 1,999 → 1,185 | 40,993,564 → 1,049,749 | 13 |
| BLD | 157 | 1,449,103,388 | 246,440,960 | 5.88× | 79,843 → 977 | 960,308,665 → 24,881,670 | 157 |
| ENV | 24 | 207,198,976 | 72,496,908 | 2.86× | 617 → 133 | 109,221,139 → 2,959,866 | 23 |
| HUM | 7 | 95,611,928 | 21,207,016 | 4.51× | 67 → 44 | 68,305,885 → 1,251,606 | 5 |
| OBS | 70 | 631,962,164 | 72,105,112 | 8.76× | 8,531 → 439 | 522,450,939 → 9,985,118 | 69 |
| RSC | 31 | 289,604,876 | 83,834,356 | 3.45× | 4,111 → 2,307 | 198,681,609 → 3,739,117 | 26 |
| TER | 55 | 434,666,864 | 94,207,128 | 4.61× | 1,506 → 132 | 301,106,403 → 7,451,875 | 55 |

Totals: **3,182,217,472 → 617,332,444 bytes (5.15×)**; draw instances **96,674 → 5,217**; embedded texture payload **2,201,068,204 → 51,319,001 bytes**.

Maximum measured world-vertex coordinate error: **0.000281612 m (0.281612 mm)**. Outputs larger than their original GLB: **0**.

Quantized asset rate: **348/359 (96.9%)**; quantized position-primitive rate: **36.7%**.

Decoded RGBA8 texture allocation proxy: **8,478,518,724 → 1,560,574,364 bytes** (sum of width × height × 4; excludes mipmaps, driver overhead and cross-asset sharing). GLB byte reduction alone is not a GPU-memory measurement.

## Method and evidence

- Exact pipeline: optimize(max_px=512) → merge_by_material → opaque_blend → compact_textures → semantic_textures(max_px=512, quality=85) → quantize_geometry.
- Every successful output is re-read and checked for glTF 2.0 magic/version/total length, JSON/BIN chunk ordering and 4-byte alignment, embedded buffer bounds, accessor strides and bounds, index ranges, acyclic scene topology, matching image MIME and payload, decodable images, and a 512-pixel texture maximum.
- Default-scene triangle counts must be identical. Every world-space vertex is compared in its material/attribute group, and geometry bounds must agree within the recorded quantization tolerance. Draw counts include mesh instances in the active scene.
- SHA256 binds original and baked bytes. ZIP extraction verifies CRC; cached NVMe files are reused only when byte length and member CRC agree. Original runtime GLBs are retained in `scene_data/assets_library/runtime_originals/`.
- Skins, animations, morphs, compressed geometry and unsupported attribute sets are preserved when the existing merge/quantization functions skip them. Per-step skip reasons remain in the manifest.
- Texture encoding is lossy and runtime-oriented; JPEG quality 85 can alter normal and metal/roughness maps. No visual equivalence, placement certification or task-level simulator acceptance is claimed from binary validation.
- Fixed compact_textures MIME mutation: a JPEG rejected for larger size must not relabel the unchanged PNG payload.

## Existing runtime assets

17/17 existing runtime GLBs baked in place; exact metrics and original hashes are recorded in `runtime_manifest.json`. Bytes **271,810,988 → 33,159,812 (8.20×)**; draw instances **6,167 → 322**; triangles **783,096**, unchanged; quantized **15/17**.

## Source completeness and limitations

All 359 delivered geometry IDs have a selected and audited runtime GLB. Pose configurations, surface effects and hidden features are separate obligations and do not create extra geometry GLBs or inflate this count; the catalog section of the manifest retains their delivered counts.

Reviewed **36** ambiguous/numeric/state/revision selections are pinned to an exact directory, archive member, byte length and CRC in `configs/drive_asset_primary_selection.json`; each includes the local package evidence used. Examples: G02U1 corrected wall UVs; G03M1/G02M1 corrected roof material factors; G05 roller-door recovery; r06_A01 container damage continuity; numbered stair/terrain components; BASE aliases for STATIC vegetation, NEUTRAL searchlights and DEPLOYED relay equipment. For the tile/wood/log components that formally share two poses, the chosen FLAT/SCATTERED library default and its pose provenance are explicit; no extra geometry or runtime acceptance is counted.

## Archives and configuration references

Verified 15 local ZIP archives with CRC checks and SHA256. Runtime configuration references now use local workspace files. Unavailable historical evidence paths: **0**. Original source locations remain in the manifest for traceability.

## Reproduction

```bash
/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python tools/stage_and_bake_drive_assets.py
/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python -m pytest -q
```

Test results are recorded after execution; this report does not infer a pass from test collection.

## Executed verification

`/mnt/data2/weizhiwei/aeroagentsim/AeroAgentSim/.venv/bin/python -m pytest -q`: **214 passed in 23.68s**.

Integration checks cover all categorized outputs and 17 runtime GLBs, exact manifest aggregates and hashes, original runtime backups, CRC-verified archive hashes, zero Drive mount references in all JSON configs, primary-selection identity pins, and byte-for-byte preservation of deformation metadata and non-image buffer views for skinned/animated/morph assets.

Captured pytest output: `scene_data/assets_library/pytest-verified.log`. No test is skipped in this staged workspace.
