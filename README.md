# RLLR -- High-level Reinforcement Control

## Pipeline

Please, refer to [Minigrid experiments](experiments/minigrid/README.md)` for more detailed description

```bash
# minigrid environment

uv run experiments/minigrid/train_similarity.py --conf conf/minigrid_zero_step_ssim.hocon

uv run experiments/minigrid/train_worker.py --conf conf/minigrid_first_step.hocon

uv run experiments/minigrid/train_master.py --conf conf/minigrid_second_step.hocon

uv run experiments/minigrid/enjoy.py --mode master


```
## Benchmarks 

| task / method                                  | PPO  | PPO + RND | PATH RL |
|------------------------------------------------|------|-----------|---------|
| MiniGrid-Dynamic-Obstacles-8x8-v0              | 0.00 | 0.61      | 0.90    |
| MiniGrid-Dynamic-Obstacles-8x8-v0 (fixed seed) | 0.00 | 0.71      |         |
| MiniGrid-PutNear-6x6-N2-v0                     | 0.40 | 0.78      |         |
| MiniGrid-PutNear-6x6-N2-v0 (fixed seed)        | 0.79 | 0.01      |         |
| MiniGrid-LavaCrossingS9N3-v0                   | 0.88 | 0.90      |         |
| MiniGrid-LavaCrossingS9N3-v0 (fixed seed)      | 0.00 | 0.91      | 0.91    |
| MiniGrid-DoorKey-8x8-v0                        | 0.98 | 0.91      |         |
| MiniGrid-DoorKey-8x8-v0 (fixed seed)           | 0.98 | 0.92      | 0.97    |
| MiniGrid-FourRooms-v0                          | 0.79 | 0.72      |         |
| MiniGrid-FourRooms-v0 (fixed seed)             | 0.83 | 0.00      |         |
| MiniGrid-KeyCorridorS3R3-v0                    | 0.92 | 0.71      |         |
| MiniGrid-KeyCorridorS3R3-v0 (fixed seed)       | 0.00 | 0.74      |         |

<p align="center">
    Table 1. MiniGrid, fully observed: Mean return over 1000 episodes
</p> 


| task / method                                  | PPO    | PPO + RND | PATH RL |
|------------------------------------------------|--------|-----------|---------|
| MiniGrid-Dynamic-Obstacles-8x8-v0              | 256    | 110.77    | 25.32   |
| MiniGrid-Dynamic-Obstacles-8x8-v0 (fixed seed) | 255.62 | 73.16     |         |
| MiniGrid-PutNear-6x6-N2-v0                     | 6.30   | 7.25      |         |
| MiniGrid-PutNear-6x6-N2-v0 (fixed seed)        | 7.00   | 27.75     |         |
| MiniGrid-LavaCrossingS9N3-v0                   | 37.36  | 35.45     |         |
| MiniGrid-LavaCrossingS9N3-v0 (fixed seed)      | 324.00 | 31.02     | 28.25   |
| MiniGrid-DoorKey-8x8-v0                        | 16.89  | 66.11     |         |
| MiniGrid-DoorKey-8x8-v0 (fixed seed)           | 12.00  | 53.78     | 18.41   |
| MiniGrid-FourRooms-v0                          | 23.21  | 30.85     |         |
| MiniGrid-FourRooms-v0 (fixed seed)             | 19.00  | 100.00    |         |
| MiniGrid-KeyCorridorS3R3-v0                    | 22.67  | 88.18     |         |
| MiniGrid-KeyCorridorS3R3-v0 (fixed seed)       | 270    | 76.95     |         |

<p align="center">
    Table 2. MiniGrid, fully observed: Mean number of steps over 1000 episodes
</p>


| task / method                                  | PPO   | PPO + RND | PATH RL |
|------------------------------------------------|-------|-----------|---------|
| MiniGrid-Dynamic-Obstacles-8x8-v0              | 0.00  | 0.97      | 0.99    |
| MiniGrid-Dynamic-Obstacles-8x8-v0 (fixed seed) | 0.00  | 0.98      |         |
| MiniGrid-PutNear-6x6-N2-v0                     | 0.50  | 0.99      |         |
| MiniGrid-PutNear-6x6-N2-v0 (fixed seed)        | 1.00  | 0.01      |         |
| MiniGrid-LavaCrossingS9N3-v0                   | 0.93  | 1.00      |         |
| MiniGrid-LavaCrossingS9N3-v0 (fixed seed)      | 0.00  | 1.00      | 0.99    |
| MiniGrid-DoorKey-8x8-v0                        | 1.00  | 1.00      |         |
| MiniGrid-DoorKey-8x8-v0 (fixed seed)           | 1.00  | 0.99      | 1.00    |
| MiniGrid-FourRooms-v0                          | 0.98  | 0.98      |         |
| MiniGrid-FourRooms-v0 (fixed seed)             | 1.00  | 0.00      |         |
| MiniGrid-KeyCorridorS3R3-v0                    | 1.00  | 0.99      |         |
| MiniGrid-KeyCorridorS3R3-v0 (fixed seed)       | 0.00  | 1.00      |         |

<p align="center">
    Table 3. MiniGrid, fully observed: Mean success rate over 1000 episodes
</p>

