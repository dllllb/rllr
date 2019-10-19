# HRL framework

The objective of HRL framework is to decompose environment state to the state of achievement of separate sub-goals. Using sub-policies allows to reuse task-specific plicies for different environments.

Sub-goals can be considered as nexus points in the state space which are commonly visited during the path to the goal. I. e. a nexus point is the part of many possible succesful trajectories of the agent while regular point is the part of small number of possible succesful trajectories.

## Framework

1. $ u_t = wm(s_t),\ \hat{u}_{t+1} = next(a_t, wm(s_t)) $
World model trasfroms state to its compact representation. World model can be trained on the task to predict $wm(s_{t+1})$ from $wm(s_t)$. Alternative: predict $wm(s_{t+k})$ from $wm(s_t)$.
Different policies $p_e$ can be used for exploration on the world model building phase
Ideally wm should factor space to the vector of independent components

2. g = pg(u_t) - generate the sequence of sub-goals (plan)

3. Select the current sub-goal g_i e. g. take the first goal of the plan

4. $ p_c = reg(u_t, g_i) $ - select a sub-policy from sub-policy registry for the current g_i

5. Use p_c to reach the goal $g_i$
p_c receives reward if $g_i$ is succesfuly reached

6. Repeat the process from step 3

## Desired state predictor

Reward estimator $r = se(s_t)$ can be trained to estimate the reward or of being in state $s_t$. Reward estimator can be trained along with the world model. Reward estimator $r = se(s_t)$ can not be directly used to predict the desired state for the current state. Hence, additional model for desired state predictor $ u_t^* = ds(u_t) $ is needed. $argmax_\Theta se(ds_\Theta(u_t))$ can be used as the objective for trainig. It is possible to train $ds$ to return the parameters of the distribution of the desired state $ f_{u^*} = ds(u_t) $ and then use $\argmax_\Theta \sum_{u \sim f_{u^*}} se(ds_\Theta(u)) $ as the objective for training.

Other possible approach to learn a model for $ds$ would be the usage of exploration history. I. e. the synyhetic task to train $ds$ would be the predicton of the state with maximum reward $u_t^*$ for every trajectory which is known from the input state. In order to iteratively increase the complexity of the task the trainig process can be started from the states near the s$u_t^*$ with gradual increase of the distance to $u_t^*$ on the state-action graph.

## Exploration policy options

$a_t = p_e(s_t)$
$p_e$ is a policy for exploration

$p_e$ options
1. Random policy
2. A policy with curiosity based reward

## Environment control learning

Environment control learning can be used as synyhetic task for trainig sub-policies

**Random position:** Train a sub-policy to reach random state $u_t'$ from $u_t$. Positive reward can be provided for the diminishing difference between $u_t'$ and $u_t$. This removes the requirement to reach the exact target state which is not always possible in changing environments. Image similarity function can be used to measure distance between $u_t'$ and $u_t$ when observed state is an image. E. g. structural similarity index (SSIM) can be used to measure distance.

**There and back:** Train a sub-policy to change a $u_t$ to $u_t'$ in a certain way and then return to the starting state $u_t$. A $u_t$ has both controllable part and non-controllable part only controllable part should be taken to the account

**Stablity task:** Try to minimize state changes

### Plan generator learning

Train $pg$ to generate a plan to reach random state $u_t'$ from $u_t$. Reward should be inversely proportional to the number of performed actions in order to reach the goal state.

Problem: some possible states are not attainable from $u_t$.
Solution 1: Perfrorm world exploration using $p_e$ and log possible state sequiences. Use initial state as $u_t$ and some state from state log as $u_t'$.
Solution 2: Use world model $wm$ and exploration policy $p_e$ to generate a $u_t'$ in simulated environment. I. e. iteratively produce next state $u_{i+1} = next(p_e(u_i), u_i)$.

## Plan-based exploration with curiosity reward

When there is a good plan generator it can be used to reach the state with good possibilities for exploration (see Go-Explore paper) and hence find the states where sparse extrinsic revard is gained
