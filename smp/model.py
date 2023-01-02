import segmentation_models_pytorch as smp


base = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=11,
)
