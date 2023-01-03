import torch.nn as nn
import segmentation_models_pytorch as smp


class DiceFocalLoss(nn.Module):
    def __init__(self, mode='multiclasss', lambda_dice=1.0, lambda_focal=1.0):
        super(DiceFocalLoss, self).__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.dice = smp.losses.DiceLoss(mode=mode)
        self.focal = smp.losses.FocalLoss(mode=mode)
    
    def forward(self, y_pred, y_true):
        dice_loss = self.dice(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss


_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'dice': smp.losses.DiceLoss,
    'focal': smp.losses.FocalLoss,
    'tversky': smp.losses.TverskyLoss,
    'jaccard': smp.losses.JaccardLoss,
    'dice_focal': DiceFocalLoss,
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        if criterion_name in ['dice', 'focal', 'tversky', 'jaccard', 'dice_focal']:
            criterion = create_fn(mode='multiclass', **kwargs)
        else:
            criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion
