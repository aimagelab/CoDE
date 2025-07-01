import numpy as np
from PIL import Image
from random import random, choice
from scipy.ndimage import gaussian_filter
import cv2
from io import BytesIO
import argparse
import torchvision.transforms as transforms
# from imwatermark import WatermarkEncoder

WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

class DataAugmentTransform:
    def __init__(self, opt):
        self.blur_prob = opt.blur_prob
        self.blur_sig = opt.blur_sig
        self.jpeg_prob = opt.jpeg_prob
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = [opt.jpeg_min, opt.jpeg_max]

    def __call__(self, img, fake=False):
        img = np.array(img)

        if random() < self.blur_prob:
            sig = sample_continuous(self.blur_sig)
            gaussian_blur(img, sig)

        if random() < self.jpeg_prob:
            method = sample_discrete(self.jpg_method)
            qual = sample_discrete(self.jpg_qual)
            img = jpeg_from_key(img, qual, method)

        return Image.fromarray(img)

# class WatermarkAddition():
#     def __init__(self, opt):
#         self.watermark = opt.watermark_prob
#         self.encoder = WatermarkEncoder()
#         self.encoder.set_watermark("bits", WATERMARK_BITS)

#     def __call__(self, img):
#         img = np.array(img)
#         if random() < self.watermark:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#             img = self.encoder.encode(img, "dwtDct")

#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)

#         return img

class PipelineAugmentation():

    def __init__(self, opt, mode='train'):
        self.data_transform = DataAugmentTransform(opt)
        # self.watermark_addition = WatermarkAddition(opt)
        self.mode = mode
        self.input_dim = opt.input_size[2]
        self.pipeline_trainf_fake = transforms.Compose([
            self.data_transform,
            transforms.RandomGrayscale(p=opt.random_grayscale_prob),
            transforms.RandomCrop(self.input_dim),
            # self.watermark_addition,
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std),
        ])

        self.pipeline_trainf_real = transforms.Compose([
            self.data_transform,
            transforms.RandomGrayscale(p=opt.random_grayscale_prob),
            transforms.RandomCrop(self.input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std),
        ])

        self.transform_test = transforms.Compose([
            transforms.CenterCrop(self.input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=opt.mean, std=opt.std),
        ])


    def __call__(self, img,fake=False):
        if self.mode == 'train' and fake:
            return self.pipeline_trainf_fake(img)
        elif self.mode == 'train' and not fake:
            return self.pipeline_trainf_real(img)
        else:
            return self.watermark_addition


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--external-transform', action='store_true', default=False)
    parser.add_argument('--blur_sig', type=float, nargs='+', default=(0, 3.))
    parser.add_argument('--blur_prob', type=float, default=0.01)
    parser.add_argument('--jpeg_prob', type=float, default=0)
    parser.add_argument('--watermark_prob', type=float, default=0.2)
    parser.add_argument('--random_grayscale_prob', type=float, default=0.01)
    parser.add_argument("--jpeg_min", default=40, type=int)
    parser.add_argument("--jpeg_max", default=100, type=int)
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N',
                       help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    opt = parser.parse_args()

    img = Image.open('/work/horizon_ria_elsa/Elsa_external_test_set/models--CompVis--stable-diffusion-v1-4/images/1094789003814.png')
    transform = PipelineAugmentation(opt)
    img = transform(img)
    img.save('test.jpg')