import torch

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

DEVICE = torch.device('cuda:3')
ENTROPY_WEIGHT = 0#0.01