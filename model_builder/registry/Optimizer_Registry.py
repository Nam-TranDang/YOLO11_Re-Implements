# Optimizer_Registry.py
from typing import Dict, Type
from .training.YOLO11_Optimizer import CosineLR, LinearLR

_Optimizers_Registry: Dict[str, Type] = {}

def Register_Optimizer(name: str):
    """Decorator for registering a scheduler class"""
    def decorator(cls):
        _Optimizers_Registry[name] = cls
        return cls
    return decorator


# These CosineLR + LinearLR are Scheduler - work with optimizer 
@Register_Optimizer("cosine_lr")
class CosineScheduler(CosineLR):
    pass

@Register_Optimizer("linear_lr")
class LinearScheduler(LinearLR):
    pass

def Get_Optimizer(name: str, **kwargs):
    """Retrieve a scheduler by name"""
    if name not in _Optimizers_Registry:
        raise ValueError(
            f"Scheduler '{name}' not found. Available: {list(_Optimizers_Registry.keys())}"
        )
    return _Optimizers_Registry[name](**kwargs)