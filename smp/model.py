from importlib import import_module
from typing import List
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders._base import EncoderMixin
from swin import SwinTransformer


def get_model_params(model_name, encoder_name, encoder_weights):
    params = {
        'encoder_name': encoder_name,
        'encoder_weights': encoder_weights,
        'classes': 11,
    }
    if model_name == 'PAN' and encoder_name[:4] == 'swin':
        params['encoder_output_stride'] = 32
    return params


def get_smp_model(model_name, encoder_name, encoder_weights):
    model_module = getattr(import_module('segmentation_models_pytorch'),
                           model_name)
    params = get_model_params(model_name, encoder_name, encoder_weights)
    model = model_module(**params)
    return model


# swin_tiny_patch4_window7_224_22kto1k_finetune
class SwinTiny(nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [96, 192, 384, 768]
        self._depth: int = 3
        self._in_channels: int = 3
        kwargs.pop('depth')
        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)


# swin_small_patch4_window7_224_22kto1k_finetune
class SwinSmall(nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [96, 192, 384, 768]
        self._depth: int = 3
        self._in_channels: int = 3
        kwargs.pop('depth')
        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)


# swin_base_patch4_window12_384_22kto1k_finetune
class SwinBase(nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [128, 256, 512, 1024]
        self._depth: int = 3
        self._in_channels: int = 3
        kwargs.pop('depth')
        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)


# swin_large_patch4_window12_384_22kto1k_finetune
class SwinLarge(nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()
        self._out_channels: List[int] = [192, 384, 768, 1536]
        self._depth: int = 3
        self._in_channels: int = 3
        kwargs.pop('depth')
        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)


# https://github.com/microsoft/Swin-Transformer/blob/main/configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml
# Swin을 smp의 encoder로 사용할 수 있게 등록
smp.encoders.encoders["swin_tiny"] = {
    "encoder": SwinTiny,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22kto1k_finetune.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 224,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        "window_size": 7,
        "drop_path_rate": 0.1,
    }
}


# https://github.com/microsoft/Swin-Transformer/blob/main/configs/swin/swin_small_patch4_window7_224_22kto1k_finetune.yaml
smp.encoders.encoders["swin_small"] = {
    "encoder": SwinSmall,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 224,
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        'num_heads': [3, 6, 12, 24],
        "window_size": 7,
        "drop_path_rate": 0.2,
    }
}


# https://github.com/microsoft/Swin-Transformer/blob/main/configs/swin/swin_base_patch4_window12_384_22kto1k_finetune.yaml
smp.encoders.encoders["swin_base"] = {
    "encoder": SwinBase,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 384,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32],
        "window_size": 12,
        "drop_path_rate": 0.2,
    }
}


# https://github.com/microsoft/Swin-Transformer/blob/main/configs/swin/swin_large_patch4_window12_384_22kto1k_finetune.yaml
smp.encoders.encoders["swin_large"] = {
    "encoder": SwinLarge,
    "pretrained_settings": {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth",
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    },
    "params": {
        "pretrain_img_size": 384,
        "embed_dim": 192,
        "depths": [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48],
        "window_size": 12,
        "drop_path_rate": 0.3,
    }
}
