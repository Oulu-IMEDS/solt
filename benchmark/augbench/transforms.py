import cv2
import Augmentor as augmentor
import albumentations as albu
from albumentations.pytorch import ToTensor
import solt
import solt.transforms as slt
from torchvision import transforms as tv_transforms


class BenchmarkTest:
    def __init__(self, img_size=256):
        self.img_size = img_size
        self.albumentations_pipeline: albu.Compose or None = None
        self.solt_pipeline: solt.Stream or None = None
        self.torchvision_pipeline: tv_transforms.Compose or None = None
        self.augmentor_pipeline: tv_transforms.Compose or None = None
        self.probablity = 0.5

    def __str__(self):
        return self.__class__.__name__

    def is_supported_by(self, library):
        if getattr(self, f"{library}_pipeline", None) is not None:
            return True
        else:
            return False

    def set_deterministic(self, val=True):
        if val:
            self.probablity = 1
        else:
            self.probablity = 0.5

    def run(self, library, imgs):
        transform = getattr(self, f"{library}_pipeline")
        for img in imgs:
            if library == "albumentations":
                transform(image=img)
            elif library == "solt":
                transform(img, return_torch=True, normalize=True)
            else:
                transform(img)


class HorizontalFlip(BenchmarkTest):
    def __init__(self, img_size=256):
        super(HorizontalFlip, self).__init__(img_size)

        self.solt_pipeline = slt.Flip(p=self.probablity, axis=1)

        self.albumentations_pipeline = albu.Compose(
            [
                albu.HorizontalFlip(p=self.probablity),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=self.probablity),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()
        _augm_ppl.flip_left_right(probability=self.probablity)
        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class VerticalFlip(BenchmarkTest):
    def __init__(self, img_size=256):
        super(VerticalFlip, self).__init__(img_size)

        self.solt_pipeline = slt.Flip(p=self.probablity, axis=0)

        self.albumentations_pipeline = albu.Compose(
            [
                albu.VerticalFlip(p=self.probablity),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomVerticalFlip(p=self.probablity),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()
        _augm_ppl.flip_top_bottom(probability=self.probablity)
        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class RotateAny(BenchmarkTest):
    def __init__(self, img_size=256):
        super(RotateAny, self).__init__(img_size)

        self.solt_pipeline = slt.Rotate(angle_range=(0, 20), p=self.probablity, padding="z")

        self.albumentations_pipeline = albu.Compose(
            [
                albu.Rotate(limit=(0, 20), p=self.probablity, border_mode=cv2.BORDER_CONSTANT, value=0),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomRotation(degrees=(0, 20), fill=0),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()
        _augm_ppl.rotate(probability=self.probablity, max_left_rotation=0, max_right_rotation=20)
        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class Crop(BenchmarkTest):
    def __str__(self, img_size=256):
        return f"{self.__class__.__name__}{self.crop_size}"

    def __init__(self, crop_size, img_size=256):
        super(Crop, self).__init__(img_size)

        self.crop_size = crop_size

        self.solt_pipeline = slt.Crop(crop_size, crop_mode="r")

        self.albumentations_pipeline = albu.Compose(
            [
                albu.RandomCrop(height=crop_size, width=crop_size),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomCrop(crop_size),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()

        _augm_ppl.crop_random(probability=1, percentage_area=crop_size / float(self.img_size))
        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class Pad(BenchmarkTest):
    def __str__(self):
        return f"{self.__class__.__name__}{self.pad}"

    def __init__(self, pad, img_size=256):
        super(Pad, self).__init__(img_size)

        self.pad = pad

        self.solt_pipeline = slt.Pad(pad)

        self.albumentations_pipeline = albu.Compose(
            [
                albu.PadIfNeeded(min_height=pad, min_width=pad, border_mode=cv2.BORDER_CONSTANT, value=0),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.Pad(pad),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class VHFlipRotateCrop(BenchmarkTest):
    def __init__(self, img_size=256):
        super(VHFlipRotateCrop, self).__init__(img_size)

        self.solt_pipeline = solt.Stream(
            [
                slt.Flip(p=self.probablity, axis=0),
                slt.Flip(p=self.probablity, axis=1),
                slt.Rotate(angle_range=(0, 20)),
                slt.Crop(224, crop_mode="r"),
            ]
        )

        self.albumentations_pipeline = albu.Compose(
            [
                albu.VerticalFlip(p=self.probablity),
                albu.HorizontalFlip(p=self.probablity),
                albu.Rotate(limit=(0, 20), p=self.probablity, border_mode=cv2.BORDER_CONSTANT, value=0),
                albu.RandomCrop(height=224, width=224),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=self.probablity),
                tv_transforms.RandomRotation(degrees=(0, 20)),
                tv_transforms.RandomCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()
        _augm_ppl.flip_top_bottom(probability=self.probablity)
        _augm_ppl.flip_left_right(probability=self.probablity)
        _augm_ppl.rotate(probability=self.probablity, max_left_rotation=0, max_right_rotation=20)
        _augm_ppl.crop_random(probability=1, percentage_area=224 / float(self.img_size))

        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class HFlipCrop(BenchmarkTest):
    def __init__(self, img_size=256):
        super(HFlipCrop, self).__init__(img_size)

        self.solt_pipeline = solt.Stream([slt.Flip(p=self.probablity, axis=1), slt.Crop(224, crop_mode="r")])

        self.albumentations_pipeline = albu.Compose(
            [
                albu.HorizontalFlip(p=self.probablity),
                albu.RandomCrop(height=224, width=224),
                ToTensor(normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}),
            ]
        )

        self.torchvision_pipeline = tv_transforms.Compose(
            [
                tv_transforms.RandomHorizontalFlip(p=self.probablity),
                tv_transforms.RandomCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        _augm_ppl = augmentor.Pipeline()
        _augm_ppl.flip_left_right(probability=self.probablity)
        _augm_ppl.crop_random(probability=1, percentage_area=224 / float(self.img_size))

        self.augmentor_pipeline = tv_transforms.Compose(
            [
                _augm_ppl.torch_transform(),
                tv_transforms.transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
