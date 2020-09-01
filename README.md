# Environment control learning

## Reach random position

Train a policy to reach random state <img src="https://render.githubusercontent.com/render/math?math=u_t^'"> from <img src="https://render.githubusercontent.com/render/math?math=u_t">. Positive reward can be provided for the diminishing difference between <img src="https://render.githubusercontent.com/render/math?math=u_t^'"> and <img src="https://render.githubusercontent.com/render/math?math=u_t">. This removes the requirement to reach the exact target state which is not always possible in changing environments. Image similarity function can be used to measure distance between <img src="https://render.githubusercontent.com/render/math?math=u_t^'"> and <img src="https://render.githubusercontent.com/render/math?math=u_t"> when observed state is an image. E. g. structural similarity index (SSIM) can be used to measure distance.

Some possible states are not attainable from <img src="https://render.githubusercontent.com/render/math?math=u_t">. The solution is to perfrorm world exploration using <img src="https://render.githubusercontent.com/render/math?math=p_e"> and log possible state sequiences. Then, initial state can be used as <img src="https://render.githubusercontent.com/render/math?math=u_t"> and some state from state log can be used as <img src="https://render.githubusercontent.com/render/math?math=u_t^'">.
