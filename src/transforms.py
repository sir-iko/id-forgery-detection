import io

from PIL import Image, ImageFilter
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class JPEGCompress:
    """Re-encode a PIL image through JPEG at a given quality to simulate compression artifacts."""

    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianBlur:
    """Apply a PIL Gaussian blur at a given sigma."""

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))


def get_transforms(split, image_size=224, jpeg_quality=None, blur_sigma=None):
    """Build image transforms for FantasyID.

    Pretrained torchvision models expect ImageNet normalisation, so we
    match that. No horizontal flip (text and faces have a canonical
    orientation), no aggressive cropping (would risk erasing the
    manipulation region). Start minimal; add deliberate augmentation
    later once we have a baseline.

    jpeg_quality and blur_sigma simulate degraded capture conditions
    (re-compression, defocus). They run at native resolution, before
    Resize, so the artifacts aren't softened by downscaling first.
    """
    base = []
    if blur_sigma is not None:
        base.append(GaussianBlur(blur_sigma))
    if jpeg_quality is not None:
        base.append(JPEGCompress(jpeg_quality))
    base += [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(base)
