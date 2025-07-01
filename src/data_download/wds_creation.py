import os
import time
import pickle
import random
import json
import tarfile
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import logging
import webdataset as wds

from download_elsa_v2 import load_all_dictionary
_logger = logging.getLogger("webdataset_creation")
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def np_images(data_dictionary, key, extension):
    img_0 = Image.open(data_dictionary[key][0]).convert('RGB')
    img_1 = Image.open(data_dictionary[key][1]).convert('RGB')
    img_2 = Image.open(data_dictionary[key][2]).convert('RGB')
    img_3 = Image.open(data_dictionary[key][3]).convert('RGB')
    img_real = Image.open(data_dictionary[key][6]).convert('RGB')

    if extension == 'jpg':
        img_0 = np.array(img_0)
        img_1 = np.array(img_1)
        img_2 = np.array(img_2)
        img_3 = np.array(img_3)
        img_real = np.array(img_real)
    return img_0, img_1, img_2, img_3, img_real

def tar_creation(data_dictionary, key, dst, dataset_name, args):
    new_img = Image.new(mode="RGB", size=(512, 512)) # PIL image
    if len(data_dictionary[key]) != 7: # filter on the dimension [7]
        return 0

    extension_gen = str(data_dictionary[key][0]).split('.')[-1]
    extension = str(data_dictionary[key][6]).split('.')[-1]
    if extension_gen != extension:
        extension = 'png'

    try:
        numpy_image_0, numpy_image_1, numpy_image_2, numpy_image_3, numpy_image_real = np_images(data_dictionary, key, extension)
    except:
        print(f'Error in reading the real image at sample ID : {key}')
        return 0

    with open(data_dictionary[key][4], 'r') as file:
        url = file.read()
    with open(data_dictionary[key][5], 'r') as file:
        cap = file.read()

    sample = {
        '__key__': dataset_name + '__' + args.split + '__' + key,
        'gen_0.jpg' if 'jpg' in extension else 'gen_0.png': numpy_image_0,
        'gen_1.jpg' if 'jpg' in extension else 'gen_1.png': numpy_image_1,
        'gen_2.jpg' if 'jpg' in extension else 'gen_2.png': numpy_image_2,
        'gen_3.jpg' if 'jpg' in extension else 'gen_3.png': numpy_image_3,
        'real.jpg' if 'jpg' in extension else 'real.png': numpy_image_real,
        'caption.txt': cap,
        'url.txt': url,
    }

    try:
        dst.write(sample)
    except:
        print(f'Error in writing the sample with ID : {key}')
        return 0


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--split', type=str, default='train')
    argparser.add_argument('--complete_dict', type=str, default='./debug/example.pkl')
    argparser.add_argument('--world_size', type=int, default=6)
    argparser.add_argument('--rank', type=int, default=0) # JOB_ARRAY
    argparser.add_argument('--dimension', type=int, default=5000)
    argparser.add_argument('--data_dir', type=Path, default='./debug')
    args = argparser.parse_args()

    dataset_name= 'elsa_v2'
    tar_path= os.path.join(args.data_dir, 'webdatasets_elsa_v2')
    os.makedirs(tar_path, exist_ok=True)
    # TODO: implement a offset for the split, for the different download split
    offset_name = 0
    start_id = (args.rank * args.dimension)
    stop_id = (args.rank * args.dimension + args.dimension)
    dst = None
    start_time = time.time()
    print(f"Start job with ID `{args.rank}` .")

    # load complete dictionary
    with open(args.complete_dict, 'rb') as f:
        data_dictionary = pickle.load(f)

    list_keys= list(data_dictionary.keys())
    # define the number of job arrays, consider the [len(list_keys)/args.dimension]
    #if args.split == 'train' or args.split == 'training': random.Random(4).shuffle(list_keys) # shuffle training set
    for i,key in tqdm(enumerate(list_keys[start_id:stop_id])):
        key

        if dst is None:
            # use offset in the tar name
            tar_shard_path = os.path.join(tar_path, dataset_name + f"-{str(args.rank+offset_name).zfill(4)}.tar") # add args.split name ?
            dst = wds.TarWriter(tar_shard_path, encoder=True)

        tar_creation(data_dictionary, key, dst, dataset_name, args)
    dst.close()

    print(f"[JOB #{args.rank}]: Created archive with {args.dimension} samples.")
    print(f"Split `{args.rank}` terminated after {args.dimension} out of {start_id - stop_id} samples.")

    print(f"--- {round((time.time() - start_time), 2)} seconds ---")
