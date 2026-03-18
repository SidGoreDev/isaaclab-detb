# Terrain Design

## DETB Terrain Taxonomy

The initial DETB terrain taxonomy should be:

- `L0_flat`
- `L1_slopes`
- `L2_rough_heightfield`
- `L3_stairs_steps`
- `L4_mixed_obstacles`
- optional `L5_adversarial_holdout`

## Design Rules

- Training and evaluation terrain sets must be distinct.
- Terrain definitions must include family, level, difficulty label, intended use, and seed policy.
- Terrain configs should map to official Isaac Lab terrain generator capabilities before custom terrain generation is introduced.
- Spawn-safe flat patches should be part of the terrain design, not an afterthought.
- Terrain metadata must propagate into manifests and downstream metrics.

## Initial Generator Mapping

Target the first DETB terrain families to official Isaac Lab terrain primitives and generator settings:

- `L0_flat`: plane or flat generated terrain
- `L1_slopes`: controlled slope or graded incline terrain
- `L2_rough_heightfield`: rough heightfield or random grid style terrain
- `L3_stairs_steps`: stairs and discrete step patterns
- `L4_mixed_obstacles`: mixed sub-terrain composition using generator proportions
- `L5_adversarial_holdout`: held-out intensified terrain used only for evaluation

## Config Strategy

Terrain config entries should eventually carry:

- `name`
- `family`
- `level`
- `difficulty_label`
- `intended_use`
- `seed`
- `cache_policy`
- `spawn_patch_policy`
- `generator_name`
- `generator_params`

## Analysis Expectations

Terrain evaluation should retain:

- per-terrain success rate
- per-terrain failure counts
- Terrain Generalization Score
- terrain battery definition used for the run

TGS is a summary only. It does not replace per-terrain evidence.
