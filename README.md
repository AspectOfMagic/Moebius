# Moebius

State-based reinforcement-learning experiments for robot manipulation. This
first milestone is a Soft Actor-Critic (SAC) baseline on one MetaWorld task.


## Included in Phase 1

- MetaWorld `MT1` wrapper with Gymnasium's `reset`/`step` API.
- Squashed Gaussian actor, twin Q critics, entropy temperature tuning, and
  Polyak-averaged target critics.
- Replay buffer, training/evaluation loop, checkpoints, and TensorBoard logs.
- Default task: `reach-v3`. Switch to another ML1 task using `--env-name`.

## Setup

Use Python 3.10+ and install the package in an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

MetaWorld is installed from its maintained Git repository, and its MuJoCo
backend may need platform-specific system libraries. Refer to the MetaWorld
installation documentation if its installation fails.

## Train

Run a short smoke experiment first:

```bash
moebius-train --total-steps 2000 --start-steps 500 --eval-interval 1000
```

Then start the standard baseline run:

```bash
moebius-train --env-name reach-v3 --total-steps 250000
```

View training and evaluation metrics:

```bash
tensorboard --logdir runs
```

## Record a trained policy

Record deterministic evaluation episodes from a saved checkpoint:

```bash
moebius-record --checkpoint checkpoints/reach-v3_sac_seed1.pt --episodes 3
```

Videos are written to `videos/`. Use `--video-dir` or `--fps` to change the
output folder or frame rate. Recording uses a top-down camera by default to
keep the object visible. Other available camera angles include:

```bash
moebius-record --checkpoint checkpoints/reach-v3_sac_seed1.pt --camera-name corner2
```

The final checkpoint is saved to `checkpoints/`. Tests that do not require a
MuJoCo installation can be run with:

```bash
pytest
```
