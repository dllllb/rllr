# Experiments

In this folder we keep the scripts of each the experiment. The subfolder `conf` contains
various configuration files used in our experiments.

## stage zero
Train a state embedding networks, capable of abstracting from irrelevant of uncontrollable
information.

```bash
train_similarity.py --conf conf/minigrid_zero_step_ssim.hocon
```

## stage 1
Train a worker agent, that dutifully carries out tasks and reaches set goals.

```bash
train_worker.py --conf conf/minigrid_first_step_ssim.hocon
```

## stage 2
Train a master agent, who sets goals for the worker.

```bash
train_master.py --conf conf/minigrid_second_step_ssim.hocon
```
