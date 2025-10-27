from typing import Dict, Type
from .training.YOLO11_Loss import ComputeLoss, FocalLoss, VFL, QFL

_Loss_Registry: Dict[str, Type] = {}

def Register_Loss(name: str):
    """Decorator for registering a loss class"""
    def decorator(cls):
        _Loss_Registry[name] = cls
        return cls
    return decorator

# This loss used for Bounding Box
@Register_Loss("compute_loss")
class Compute_YOLO11_Loss(ComputeLoss):
    pass

# Below Are 3 loss for Classification task
@Register_Loss("focal_loss")
class Focal_Loss(FocalLoss):
    pass


@Register_Loss("vfl")
class VFL_Loss(VFL):
    pass


@Register_Loss("qfl")
class QFL_Loss(QFL):
    pass


def Get_Loss(name: str,  **kwargs):
    """Retrieve a loss by name"""
    if name not in _Loss_Registry:
        raise ValueError(
            f"Loss '{name}' not found. Available: {list(_Loss_Registry.keys())}"
        )
    return _Loss_Registry[name](**kwargs)