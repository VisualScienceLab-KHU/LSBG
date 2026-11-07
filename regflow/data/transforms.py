import albumentations as albu
import numpy as np
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class SampleTransform:
    def __init__(self, image_size=512, augmentation=False, recon_size=128):
        if image_size <= 0 or recon_size <= 0:
            raise ValueError("image_size and recon_size must be positive.")
        self.image_size = int(image_size)
        self.recon_size = int(recon_size)
        self.augment = albu.Compose(
            [
                albu.RandomScale(scale_limit=0.2, p=1),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=0.1, p=1
                ),
                albu.ImageCompression(quality_lower=70, quality_upper=100, p=0.2),
                albu.RandomRotate90(p=0.5),
                albu.GaussianBlur(blur_limit=(3, 7), p=0.2),
            ]
            if augmentation
            else []
        )
        self.resize = albu.Compose(
            [
                albu.Resize(self.image_size, self.image_size),
                albu.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                ToTensorV2(transpose_mask=True),
            ]
        )

    def __call__(self, image, recon_map, mask=None):
        image = np.asarray(image.convert("RGB"))
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("Image dimensions must be nonzero.")
        recon_dtype = recon_map.dtype
        recon = recon_map.float()
        if recon.shape[-2:] != image.shape[:2]:
            recon = F.interpolate(
                recon.unsqueeze(0), size=image.shape[:2], mode="bilinear", align_corners=False
            ).squeeze(0)
        recon = recon.to(recon_dtype).squeeze(0).cpu().float().numpy()
        masks = [recon]
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != image.shape[:2]:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image shape {image.shape[:2]}."
                )
            masks.insert(0, mask.astype(np.float32))
        transformed = self.augment(image=image, masks=masks)
        transformed = self.resize(**transformed)
        recon = transformed["masks"][-1].unsqueeze(0).float()
        if recon.shape[-2:] != (self.recon_size, self.recon_size):
            recon = F.interpolate(
                recon.unsqueeze(0),
                size=(self.recon_size, self.recon_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        result = {"image": transformed["image"], "recon_map": recon.to(recon_dtype)}
        if mask is not None:
            result["mask"] = transformed["masks"][0].unsqueeze(0).float()
        return result
