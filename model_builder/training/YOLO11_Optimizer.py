import math
import copy
import torch
import numpy as np

# Both CosineLR + LinearLR are scheduler 
class CosineLR:
    """Cosine annealing learning rate scheduler with warmup"""
    def __init__(self, epochs, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps)

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    """Linear warmup + decay learning rate scheduler"""
    def __init__(self, epochs, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(epochs * num_steps - warmup_steps)

        warmup_lr = np.linspace(min_lr, max_lr, warmup_steps, endpoint=False)
        decay_lr = np.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = np.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]

# Schedulers → control learning rate per step.
# EMA → controls how model weights are averaged over time.
'''
    Optimizer → moves parameters toward minimizing loss (noisy updates).
    EMA → smooths the optimizer’s noisy trajectory for better stability and generalization.
    ==> EMA help to stablize in training 
    
    Context: Optimizer can have loss value, chart could esclated by Loss.
    + EMA runs in parallel and smooths the optimizer updates (smooth the training curve)
'''
class EMA:
    """
    Exponential Moving Average (EMA) for model weights.
    Keeps a smoothed copy of model parameters to stabilize training.
    """
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Clone the model for maintaining an exponential moving average
        self.ema = copy.deepcopy(model).eval()
        self.updates = updates
        # Decay ramp-up: helps stabilize EMA in early training
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA weights using current model weights"""
        if hasattr(model, 'module'):
            model = model.module

        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            model_state = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * model_state[k].detach()


def set_params(model, decay):
    """Group model parameters for weight decay optimization"""
    p1, p2 = [], []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)

    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if not p.requires_grad:
                continue
            if n == "bias" or (n == "weight" and isinstance(m, norm)):
                p1.append(p)  # no weight decay
            else:
                p2.append(p)  # apply weight decay

    return [
        {'params': p1, 'weight_decay': 0.00},
        {'params': p2, 'weight_decay': decay}
    ]


def clip_gradients(model, max_norm=10.0):
    """Gradient clipping for training stability"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)