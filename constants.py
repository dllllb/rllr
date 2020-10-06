import torch
import math

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

    def __init__(self, val, steps_per_epoch):
        self.step = 0
        self.val = val
        self.steps_per_epoch = steps_per_epoch

    def __call__(self):
        temperature = math.exp(-1*self.step/(EPOCHS*self.steps_per_epoch))
        self.step += 1
        return self.val * temperature

#DEVICE = torch.device('cuda:2')
DEVICE = torch.device('cpu')
EPOCHS = 1000
ENTROPY_WEIGHT =  ExponentialWeighter(0.75, steps_per_epoch=30)# 0.75# 0.65  


