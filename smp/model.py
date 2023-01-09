from importlib import import_module


def get_smp_model(model_name, encoder_name, encoder_weights):
    module = getattr(import_module("segmentation_models_pytorch"),
                     model_name)
    model = module(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=11,
    )
    return model
