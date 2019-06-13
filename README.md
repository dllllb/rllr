# HRL framework

- sub-goals can be considered as nexus points in the state space which are commonly visited during the path to the goal
  - i. e. a nexus point is the part of many possible succesful trajectories of agent while regular point is the part of small number of possible succesful trajectories

- decompose environment state to the state of achievement of separate sub-goals

- using sub-policies allows to reuse task-specific plicies for different environments

### Framework

1. $ u_t = wm(s_t),\ \hat{u}_{t+1} = next(a_t, wm(s_t)) $
    - wm trasfroms state to its compact representation
      - wm can be trained on the task to predict $wm(s_{t+1})$ from $wm(s_t)$
        - alternative: predict $wm(s_{t+k})$ from $wm(s_t)$
      - different policies can be used for exploration on world model building phase
    - ideally wm should factor space to the vector of independent components
2. $r = se(s_t)$
    - state estimator which estimates the reward of being it state $s_t$
    - reward estimator can be trained along with the world model
3. g = pg(u_t)
    - generate a sequence of sub-goals (plan)
4. for each g_i in g repeat the next steps
5. $ p_c = reg(u_t, g_i) $
  - select a sub-policy from sub-policy registry
6. use p_c to reach the goal $g_i$
    - p_c receives reward if $g_i$ is succesfuly reached

#### Desired state distribution plan generator

1. $ d = des(u_t) $
    - for the current state $u_t$ predict desired state values distribution
      - each state component can have its own distribution
      - uniform distribution with bounds would define range of possible values
        - uniform distribution with loose (poossibly infinite) bounds mark the particular state component as not significant
        - uniform distribution with narrow bounds would mark exact value of the particular state component as very significant
    - use reward estimator *se* to train the desired state producer *des*
2. for each component of d produce $P_{d_i}(u_{t_i}) > \epsilon$ as a plan item $g_i: (i, d_i)$
    - $\epsilon$ is a hyperparameter and should be a reasonably high value e. g. 0.5
    - i can be choosen randomly
      - if $g_i$ is already satisfied for the current state then i-th component should be ignored and no plan item should be generated

### Exploration policy options

- $a_t = p_e(s_t)$
  - $p_e$: a policy for exploration

- $p_e$ options
  - random policy
  - a policy with curiosity based reward

### Environment control learning

- sub-policy learning tasks
  - random pos
    - train a sub-policy to reach random state $u_t'$ from $u_t$
      - positive reward can be provided for the diminishing difference between $u_t'$ and $u_t$
        - this removes the requirement to reach the exact target state which is not always possible in changing environments
        - image similarity function can be used to measure distance between $u_t'$ and $u_t$ when observed state is an image
          - e. g. structural similarity index (SSIM) can be used to measure distance
  - there and back
    - train a sub-policy to change a $u_t$ to $u_t'$ in a certain way and then return to the starting state $u_t$
      - a $u_t$ has both controllable part and non-controllable part only controllable part should be taken to the account
  - stablity task
    - try to minimize state changes

### Plan generator learning

- train pg to generate a plan to reach random state $u_t'$ from $u_t$
  - reward should be inversely proportional to the number of performed actions in order to reach the goal state

- problem: some possible states are not attainable from $u_t$
  - solution 1:
    - perfrorm world exploration using $p_e$ and log possible state sequiences
    - use initial state as $u_t$ and some state from state log as $u_t'$
  - solution 2:
    - use world model $wm$ and exploration policy $p_e$ to generate a $u_t'$ in simulated environment
      - i. e. iteratively produce next state $u_{i+1} = next(p_e(u_i), u_i)$

## Plan-based exploration with curiosity reward

- when there is a good plan generator it can be used to reach the state with good possibilities for exploration (see Go-Explore paper)

# Imaginary world planning

- use Monte-Carlo Tree Search to find effective trajectory in imaginary world based on world model
  - i. e. mark trajectory elements according to the sum of imaginary rewards from the word model
    - use sum of node rewards in bandits algorithm to select the next action for the explored trajectory
