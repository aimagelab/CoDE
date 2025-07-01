import torch
import torchvision.transforms as transforms
import augly.image as imaugs
from PIL import Image
import random
import math
def custom_range(min, max, steps,):
    results = torch.linspace(min, max, steps)
    return results   # float

class MassiveTransform(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        # val_jitter = custom_range(args.jitter_min, args.jitter_max)
        #COLOR_JITTER_PARAMS = {
        #    "brightness_factor": val_jitter,
        #    "contrast_factor": val_jitter,
        #    "saturation_factor": val_jitter, }
        # val_crop = custom_range(args.crop_min, args.crop_max)

        transform = []
        #transform.append(imaugs.RandomEmojiOverlay()) # very difficult to deal with maybe we should suspend (a lot of parameter)
        #transform.append(imaugs.MemeFormat(text='hello world'))
        transform.append(imaugs.RandomBlur(min_radius=args.blur_min, max_radius=args.blur_max))
        transform.append(imaugs.RandomBrightness(min_factor =args.brightness_min, max_factor =args.brightness_max))
        transform.append(imaugs.RandomAspectRatio(min_ratio=args.ratio_min, max_ratio = args.ratio_max))
        transform.append(imaugs.RandomPixelization(min_ratio=args.pix_min, max_ratio= args.pix_max))
        transform.append(imaugs.RandomRotation(min_degrees=args.rotatio_min, max_degrees=args.rotatio_max))

        c_range = custom_range(args.contrast_min, args.contrast_max, args.step)
        transform.append([imaugs.Contrast(factor=item.item()) for item in c_range])  # factor:float
        c_range = custom_range(args.saturation_min, args.saturation_max, args.step)
        transform.append([imaugs.Saturation(
            factor=item.item()) for item in c_range])  # factor:float [0.5 - 1.5]

        c_range = custom_range(args.jpeg_min, args.jpeg_max, args.step)
        transform.append([imaugs.EncodingQuality(
            quality=int(item.item())) for item in c_range])  # quality:int [0 - 100] jpeg compression

        transform.append(imaugs.Grayscale(mode='luminosity'))
        transform.append(imaugs.HFlip())

        c_range = custom_range(args.opacity_min, args.opacity_max, args.step)
        transform.append([imaugs.Opacity(level=item.item()) for item in c_range])  # level:float [0 - 1]

        c_range = custom_range(args.overlay_min, args.overlay_max, args.step)
        transform.append([imaugs.OverlayStripes(line_width=item.item()) for item in c_range])
        c_range = custom_range(args.pad_min, args.pad_max, args.step)
        transform.append([imaugs.Pad(w_factor=item.item(), h_factor=item.item()) for item in c_range])

        c_range = custom_range(args.resize_min, args.resize_max, args.step)
        transform.append([imaugs.Resize(width=int(item.item()), height=int(item.item()), resample=Image.BICUBIC) for item in c_range])

        c_range = custom_range(args.crop_min, args.crop_max, args.step)
        transform.append([transforms.RandomCrop(
            size=item.item(), pad_if_needed=True) for item in c_range])

        c_range = custom_range(args.scale_min, args.scale_max, args.step)
        transform.append([
            imaugs.Scale(factor=item.item()) for item in c_range])  # scale_factor:float [0.5 - 1.5]

        c_range = custom_range(args.sharp_min, args.sharp_max, args.step)
        transform.append([imaugs.Sharpen(factor=item.item()) for item in c_range])

        c_range = custom_range(args.shuffle_min, args.shuffle_max, args.step)
        transform.append([imaugs.ShufflePixels(factor=item.item(), seed=args.seed) for item in c_range] )

        c_range = custom_range(args.skew_min, args.skew_max, args.step)
        transform.append([
            imaugs.Skew(skew_factor=item.item()) for item in c_range]) # skew_factor:float [-1 - 1]
        self.global_view_dino = transforms.RandomResizedCrop(224, scale=args.global_crops_scale, interpolation=Image.BICUBIC)
        self.local_view_dino = transforms.RandomResizedCrop(224, scale=args.local_crops_scale, interpolation=Image.BICUBIC)
        self.transform = transform
        self.tot_transforms = len(self.transform)
        self.num_transform = args.num_transform
        self.mean = args.mean
        self.std = args.std
        self.input_dim = args.input_size[2]
        self.last_crop = args.last_crop
        self.crop_pct = args.crop_pct
        self.random_crop = args.random_crop
        self.mode = None
        if self.num_transform > self.tot_transforms: raise ValueError(
            'Number of transforms to apply is greater than the total number of transforms available')


    def forward(self, img: Image, transform=True, dino='') -> Image:
        num_transform = random.sample(range(self.num_transform+1), 1)[0]
        list_transform = sorted(random.sample(range(self.tot_transforms), num_transform))
        composed_transform = []
        if dino == 'global':
            composed_transform.append(self.global_view_dino)
        elif dino == 'local':
            composed_transform.append(self.local_view_dino)
        if transform:
            # composed_transform = [None] * len(list_transform)
            for i, el in enumerate(list_transform):
                composed_transform.append(self.transform[el] if not type(self.transform[el]) is list else random.sample(self.transform[el],1)[0])
        if self.random_crop and self.mode == 'train':
            composed_transform.append(transforms.RandomCrop(self.input_dim, pad_if_needed=True, fill=255, padding_mode='constant')) # white padding
        elif self.random_crop and self.mode == 'test':
            composed_transform.append(transforms.CenterCrop(self.input_dim))
        elif self.last_crop:
            composed_transform.append(
                transforms.Resize(math.floor(self.input_dim/self.crop_pct), interpolation=transforms.InterpolationMode.BICUBIC))
            composed_transform.append(transforms.CenterCrop(self.input_dim))
        else:
            composed_transform.append(transforms.Resize((self.input_dim, self.input_dim), interpolation=transforms.InterpolationMode.BICUBIC))
        composed_transform.append(transforms.ToTensor())
        composed_transform.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(composed_transform)(img)

class LowTransform(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        val_jitter = custom_range(args.jitter_min, args.jitter_max)
        COLOR_JITTER_PARAMS = {
            "brightness_factor": val_jitter,
            "contrast_factor": val_jitter,
            "saturation_factor": val_jitter, }
        # val_crop = custom_range(args.crop_min, args.crop_max)

        transform = []
        transform.append(imaugs.RandomBlur())
        transform.append(imaugs.RandomBrightness())
        transform.append(imaugs.RandomAspectRatio())
        transform.append(imaugs.RandomEmojiOverlay())
        transform.append(imaugs.RandomPixelization())  # normal
        transform.append(imaugs.RandomRotation(min_degrees=90.0, max_degrees=270.0))

        transform.append(imaugs.ColorJitter(**COLOR_JITTER_PARAMS))
        transform.append(imaugs.Contrast(factor=custom_range(args.contrast_min, args.contrast_max)))  # factor:float
        transform.append(imaugs.Saturation(
            factor=custom_range(args.saturation_min, args.saturation_max)))  # factor:float [0.5 - 1.5]
        transform.append(imaugs.Crop(x1=0.25, y1=0.25, x2=0.75, y2=0.75))  # FIXME: fx dimension for the crop
        transform.append(imaugs.EncodingQuality(
            quality=int(custom_range(args.jpeg_min, args.jpeg_max))))  # quality:int [0 - 100] jpeg compression
        transform.append(imaugs.Grayscale(mode='luminosity'))
        transform.append(imaugs.HFlip())
        transform.append(imaugs.MemeFormat(text='hello world'))

        transform.append(imaugs.Opacity(level=custom_range(args.opacity_min, args.opacity_max)))  # level:float [0 - 1]
        transform.append(imaugs.OverlayStripes(line_width=0.5))
        transform.append(imaugs.Pad(w_factor=0.25, h_factor=0.25))  # w_factor:float [0 - 1], h_factor:float [0 - 1]

        # size= int(custom_range(args.resize_min, args.resize_max)) # FIXME: at the end images with different dimensions
        # transform.append(imaugs.Resize(width=size, height=size, resample=Image.BICUBIC))

        transform.append(
            imaugs.Scale(factor=custom_range(args.scale_min, args.scale_max)))  # scale_factor:float [0.5 - 1.5]
        transform.append(imaugs.Sharpen(factor=custom_range(args.sharp_min, args.sharp_max)))
        transform.append(imaugs.ShufflePixels(factor=custom_range(args.shuffle_min, args.shuffle_max), seed=42))
        transform.append(
            imaugs.Skew(skew_factor=custom_range(args.skew_min, args.skew_max))) # skew_factor:float [-1 - 1]

        self.transform = transform
        self.tot_transforms = len(self.transform)
        self.num_transform = args.num_transform
        self.mean = args.mean
        self.std = args.std
        if self.num_transform > self.tot_transforms: raise ValueError(
            'Number of transforms to apply is greater than the total number of transforms available')
        self.compose_transform()

    def forward(self, img: Image, transform=True) -> Image:
        list_transform = random.sample(range(self.tot_transforms),1)[0]
        if transform:
            #print(self.transform[list_transform])
            composed_transform = self.transform[list_transform]
            return composed_transform(img)
        return self.transform[-1](img)

    def compose_transform(self):
        for idx, element in enumerate(self.transform):
            composed_transform = [element]
            composed_transform.append(
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC))
            composed_transform.append(transforms.ToTensor())
            composed_transform.append(transforms.Normalize(self.mean, self.std))
            self.transform[idx] = transforms.Compose(composed_transform)
        # the last element is the transform without augmentations
        composed_transform = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(self.mean, self.std)]
        self.transform.append(transforms.Compose(composed_transform))