import argparse
import os
import sys
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from PIL import Image
import pickle
from tqdm import tqdm
from code_model import VITContrastiveHF
from dataset_paths_d3 import DATASET_PATHS_D3

import random


def set_seed(SEED=0):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc


def calculate_acc_svm(y_true, y_pred):
    y_true[y_true == 1] = -1
    y_true[y_true == 0] = 1
    r_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1])  # > thres)
    f_acc = accuracy_score(y_true[y_true == -1], y_pred[y_true == -1])  # > thres)
    acc = accuracy_score(y_true, y_pred)  # > thres)
    return r_acc, f_acc, acc


@torch.inference_mode()
def validate(model, loader, opt=None):
    with torch.no_grad():
        y_true, y_pred = [], []
        print("Length of dataset: %d" % (len(loader)))
        for img, label, img_paths in tqdm(loader):
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).flatten().tolist())
            y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    threshold = 0.5
    if model.classificator_type == "linear" or model.classificator_type == "knn":
        r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, threshold)
    else:
        r_acc0, f_acc0, acc0 = calculate_acc_svm(y_true, y_pred)

    return r_acc0, f_acc0, acc0


# = = = = = = = = = = = = =
# = = = = = = = = = = = = = = = = = = = = = = = #


def recursively_read(
    rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp", "JPG"]
):
    out = []
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split(".")[1] in exts) and (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=""):
    if ".pickle" in path:
        with open(path, "rb") as f:
            image_list = pickle.load(f)
        image_list = [item for item in image_list if must_contain in item]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(
        self,
        real_path,
        fake_path,
        data_mode,
        transform,
    ):

        assert data_mode in ["wang2020", "ours"]
        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list

        # = = = = = =  label = = = = = = = = = #

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
        self.transform = transform

    def read_path(self, real_path, fake_path, data_mode):

        if data_mode == "wang2020":
            real_list = get_list(real_path, must_contain="0_real")
            fake_list = get_list(fake_path, must_contain="1_fake")
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        # assert len(real_list) == len(fake_list)
        print("real: %d, fake: %d" % (len(real_list), len(fake_list)))
        return real_list, fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]

        img = Image.open(img_path).convert("RGB")

        img = self.transform(img)
        return img, label, img_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--real_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument(
        "--fake_path", type=str, default=None, help="dir name or a pickle"
    )
    parser.add_argument("--data_mode", type=str, default=None, help="wang2020 or ours")
    parser.add_argument("--result_folder", type=str, default="./results", help="")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--classificator_type", type=str, default="linear")
    opt = parser.parse_args()

    os.makedirs(opt.result_folder, exist_ok=True)

    model = VITContrastiveHF(classificator_type=opt.classificator_type)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    print("Model loaded..")
    model.eval()
    model.cuda()
    dataset_paths = DATASET_PATHS_D3
    for dataset_path in dataset_paths:
        set_seed()
        print(f"dataset_path: {dataset_path}")
        dataset = RealFakeDataset(
            dataset_path["real_path"],
            dataset_path["fake_path"],
            dataset_path["data_mode"],
            transform=transform,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
        )
        r_acc0, f_acc0, acc0 = validate(model, loader, opt=opt)

        with open(os.path.join(opt.result_folder, "acc0.txt"), "a") as f:
            f.write(
                dataset_path["key"]
                + ": "
                + str(round(r_acc0 * 100, 2))
                + "  "
                + str(round(f_acc0 * 100, 2))
                + "  "
                + str(round(acc0 * 100, 2))
                + "\n"
            )
