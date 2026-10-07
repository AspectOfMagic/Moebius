# Moebius

Moebius is a state-based reinforcement-learning project for MetaWorld robot
manipulation. It supports a single-task Soft Actor-Critic (SAC) baseline and a
PEARL agent that adapts across object shapes, environments, and optional hidden
object dynamics.

## Setup

Use Python 3.10 or newer and install the project in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
```

MetaWorld uses MuJoCo and may require platform-specific rendering libraries.
Run the test suite after installation:

```bash
pytest
```

## SAC training

### Object reaching with the original SAC protocol

This is the pure SAC baseline. It trains on `reach-v3` without demonstrations
or behavior cloning.

```bash
moebius-train --algorithm sac --env-name reach-v3 \
  --demo-episodes 0 --total-steps 1000000 \
  --eval-interval 10000 \
  --log-dir runs/sac_reach \
  --checkpoint-dir checkpoints/sac_reach
```

### Pick-and-place with demonstrations

This run collects successful scripted-expert demonstrations, behavior-clones
the actor before training, and keeps demonstrations in the online SAC updates
while the cloning weight decays.

```bash
moebius-train --algorithm sac --env-name pick-place-v3 \
  --demo-episodes 100 --bc-pretrain-updates 10000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 \
  --bc-decay-steps 250000 --total-steps 1000000 \
  --eval-interval 10000 \
  --log-dir runs/sac_pick_place \
  --checkpoint-dir checkpoints/sac_pick_place
```

## PEARL training

### Object shapes across multiple MetaWorld environments

This is the default shape-focused PEARL experiment. One agent trains across
four related manipulation environments and three object shapes. Mass and
friction remain at MetaWorld's nominal values, force control is disabled, and
each environment uses its dedicated scripted expert.

```bash
moebius-train --algorithm pearl --pearl-task-distribution dynamics \
  --pearl-env-names pick-place-v3 pick-place-wall-v3 shelf-place-v3 pick-out-of-hole-v3 \
  --object-shapes cylinder box bar \
  --num-train-tasks 20 --num-eval-tasks 10 \
  --latent-dim 5 --context-size 64 --meta-batch-size 8 \
  --pearl-demo-episodes-per-task 5 \
  --pearl-bc-pretrain-updates 5000 \
  --pearl-critic-pretrain-updates 5000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 \
  --bc-decay-steps 500000 --pearl-bc-weight-floor 0.1 \
  --context-sequences-per-task 4 \
  --pearl-reward-scale 0.1 --pearl-initial-alpha 0.02 \
  --pearl-decoder-weight 0 --max-episode-steps 300 \
  --pearl-eval-sequences-per-task 3 --eval-interval 30000 \
  --total-steps 1000000 \
  --log-dir runs/pearl_multi_shape \
  --checkpoint-dir checkpoints/pearl_multi_shape
```

The environment identity is not added directly to the observation. PEARL must
infer useful task information from transition context. Mass and friction
randomization remain disabled because `--pearl-vary-object-dynamics` is not
present.

### Object shapes with hidden mass, friction, and grip force

This experiment uses `pick-place-v3` because force control and contact-force
observations currently support that environment only. Every task varies object
shape, size, mass, and friction. Before contact, the environment closes the
fingers at a fixed speed; after contact, the fourth action selects squeeze
force. The reward penalizes force above the task's calibrated requirement.

```bash
moebius-train --algorithm pearl --pearl-task-distribution dynamics \
  --pearl-env-names pick-place-v3 \
  --object-shapes cylinder box bar \
  --pearl-vary-object-dynamics \
  --gripper-force-limit 60 --gripper-force-control \
  --contact-observations --pearl-grip-penalty 1.0 \
  --pearl-expert-variants careful slow_transport low_grasp metaworld \
  --num-train-tasks 20 --num-eval-tasks 10 \
  --latent-dim 5 --context-size 64 --meta-batch-size 8 \
  --pearl-demo-episodes-per-task 5 \
  --pearl-bc-pretrain-updates 5000 \
  --pearl-critic-pretrain-updates 5000 \
  --demo-batch-fraction 0.25 --bc-weight 1.0 \
  --bc-decay-steps 500000 --pearl-bc-weight-floor 0.1 \
  --context-sequences-per-task 4 \
  --pearl-reward-scale 0.1 --pearl-initial-alpha 0.02 \
  --pearl-decoder-weight 0 --max-episode-steps 300 \
  --pearl-eval-sequences-per-task 3 --eval-interval 30000 \
  --total-steps 1000000 \
  --log-dir runs/pearl_hidden_dynamics_force \
  --checkpoint-dir checkpoints/pearl_hidden_dynamics_force
```

During setup, candidates are assigned balanced gentle, medium, and strong grip
bands. The expert strategies are tried in the listed order, and a task is kept
only when one strategy succeeds reliably within its assigned band. Automatic
pre-contact closing is not penalized because that force was not selected by
the agent.

Training metrics can be viewed with:

```bash
tensorboard --logdir runs
```

For PEARL, the main adaptation signals are `eval/adaptation_gain`,
`stats/z_mean_task_std`, and the `eval/success_after_context_*` curves.

## Record trained policies

Record three deterministic SAC episodes:

```bash
moebius-record \
  --checkpoint checkpoints/sac_pick_place/pick-place-v3_sac_seed1.pt \
  --episodes 3
```

Record three PEARL context episodes followed by one adapted rollout on a
held-out task:

```bash
moebius-record \
  --checkpoint checkpoints/pearl_multi_shape/multi-pick-place_pearl_seed1.pt \
  --task-index 0 --adaptation-episodes 3
```

Videos are saved under `videos/`. Use `--video-dir`, `--fps`, or
`--camera-name` to change the output directory, frame rate, or camera.
