# Terrain Research

This note captures the research basis for DETB terrain design. The goal is to use Isaac Lab terrain capabilities and established quadruped-locomotion patterns rather than inventing terrain logic ad hoc.

## Primary Sources

- Isaac Lab terrain generator implementation: https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab/terrains/terrain_generator.html
- Isaac Lab terrain API docs: https://docs.robotsfan.com/isaaclab_official/v2.3.1/source/api/lab/isaaclab.terrains.html
- legged_gym repository: https://github.com/leggedrobotics/legged_gym
- Unitree Isaac Lab repository: https://github.com/unitreerobotics/unitree_sim_isaaclab
- Rapid locomotion on challenging terrain: https://doi.org/10.1126/scirobotics.abc5986
- Robust Rough-Terrain Locomotion with a Quadrupedal Robot: https://doi.org/10.1109/ICRA.2018.8460731
- Terrain-Aware Quadrupedal Locomotion via Reinforcement Learning: https://arxiv.org/abs/2310.04675
- Scaling Rough Terrain Locomotion With Automatic Curriculum RL: https://arxiv.org/abs/2601.17428

## Isaac Lab Findings

- Isaac Lab already provides a terrain generator abstraction with explicit configuration, terrain mixing, and generator-level controls.
- Terrain difficulty should be treated as a controlled input, not an informal label.
- Terrain generation cost and mesh density are real performance considerations and should be exposed in config, not hidden.
- Spawn-safe flat patches are important for reset stability and fair evaluation.
- Terrain seeds and cache behavior matter for reproducibility and should be explicit in run artifacts.

## Repo Pattern Findings

- `legged_gym` remains a useful reference for rough-terrain curriculum structure and how to progressively raise terrain challenge during training.
- `unitree_sim_isaaclab` is a useful reference for organizing locomotion tasks and simulator-backed environments in Isaac Lab rather than mixing runtime concerns into analysis code.
- Public locomotion repos tend to separate terrain families such as slopes, stairs, rough fields, and obstacles instead of treating all rough terrain as one bucket.

## Research Implications For DETB

- DETB should keep training terrains and held-out evaluation terrains distinct.
- Terrain studies should preserve per-terrain results and failure labels instead of collapsing everything into one scalar.
- Sensor trade studies should be terrain-conditional because terrain-aware perception matters most on stairs, discrete obstacles, and rough heightfields.
- Terrain taxonomy should be explicit enough to support later requirement statements.

## DETB Terrain Best Practices

- Keep terrain family and difficulty metadata in config.
- Keep terrain seeds explicit and reviewable.
- Preserve flat patches for reset and spawn safety.
- Start simple and add terrain richness only after baseline locomotion is stable.
- Use TGS only as a summary metric and always preserve per-terrain curves.
- Keep terrain generation decisions documented so they can survive simulator updates.
