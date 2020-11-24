import torch
import math
import spinup

def is_nan(val, desc):
    if val is None:
        #print(f'\nVAL IS NONE {desc}\n')
        return False

    if torch.isnan(val).any().item():
        print(f'+++++++++++++++++nan in {desc} +++++++++++++++++++\n')
        return True

    if torch.isinf(val).any().item():
        print(f'+++++++++++++++++inf in {desc} +++++++++++++++++++\n')
        return True

    return False

class Hook():
    def __init__(self, module, desc, backward=False):
        self.backforward = backward
        if backward==False:
            self.desc = f'forward {desc}'
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.desc = f'backforward {desc}'
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        for i, inp in enumerate(input):
            is_nan(inp, f'{self.desc} input_{i+1}')
        for i, ot in enumerate(output):
            is_nan(ot, f'{self.desc} output_{i+1}')

    def close(self):
        self.hook.remove()

class ExponentialWeighter:

    def __init__(self, val, min_val, steps_per_epoch, epochs):
        self.iter = 0
        self.val = val
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_val = min_val

    def __call__(self):
        #temperature = math.exp(-1*self.iter/(EPOCHS*self.steps_per_epoch))
        temperature = math.exp(-self.iter/(self.epochs*self.steps_per_epoch))
        return max(self.val * temperature, self.min_val)

    def step(self):
        self.iter += 1

DEVICE = torch.device('cuda:1')
#DEVICE = torch.device('cpu')
EPOCHS = 1000
BATCH_SIZE = 50#300#50
L_RATE = 1e-3# 1e-3
ENTROPY = 1# start 0.9 mean possible entropy bp 0.05 multiplier 50
'''
rather good values
L_RATE = 1e-3
ENTROPY = 0.9# start 0.9 mean possible entropy bp 0.05 multiplier 50
'''
#ENTROPY_WEIGHT =  ExponentialWeighter(ENTROPY, steps_per_epoch=BATCH_SIZE, epochs=EPOCHS)# 0.75# 0.65  
ENTROPY_WEIGHT =  ExponentialWeighter(ENTROPY, min_val=0, steps_per_epoch=1, epochs=100)# 0.75# 0.65  
TASKS_FILE = 'tasks.pkl'
CIRCULUM_LEARNING = True


