from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split, image_size=224):
    """Build image transforms for FantasyID.

    Pretrained torchvision models expect ImageNet normalisation, so we
    match that. No horizontal flip (text and faces have a canonical
    orientation), no aggressive cropping (would risk erasing the
    manipulation region). Start minimal; add deliberate augmentation
    later once we have a baseline.
    """
    base = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)
