# Experiments

In this folder we keep the scripts of each the experiment. The subfolder `conf` contains
various configuration files used in our experiments.

## stage zero
Train a state embedding networks, capable of abstracting from irrelevant of uncontrollable
information.

```bash
python train_state_distance_network.py --conf conf/minigrid_zero_step.hocon
```

## stage 1
Train a worker agent, that dutifully carries out tasks and reaches set goals.

```bash
python train_worker.py --conf conf/minigrid_first_step.hocon
```

## stage 2
Train a master agent, who sets goals for the worker.

```bash
python train_master.py --conf conf/minigrid_second_step.hocon
```