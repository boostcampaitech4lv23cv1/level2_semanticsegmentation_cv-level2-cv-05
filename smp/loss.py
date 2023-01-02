import torch.nn as nn
import segmentation_models_pytorch as smp


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'dice': smp.losses.DiceLoss,
    'focal': smp.losses.FocalLoss,
    'tversky': smp.losses.TverskyLoss,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        if criterion_name in ['dice', 'focal', 'tversky']:
            criterion = create_fn(mode='multiclass', **kwargs)
        else:
            criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
