import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import datasets
from datasets import load_dataset, Image
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import pickle

from local_utils.logger import custom_setup_default_logging
_logger = logging.getLogger("datasetdownload")

def data_donwload(args):
    '''
    This function download the dataset and split it in the different ranks
    '''
    data_dictionary = defaultdict(list)
    idx_directory = 0
    number_of_samples = 0
    path_rank = args.data_dir / f"data_d3/job-{args.rank + args.offest_rank_folder}"
    incomplete_images = []
    if not os.path.exists(path_rank):
        os.makedirs(path_rank)
    faulted_parquet = []
    custom_setup_default_logging(log_path=args.data_dir / path_rank / f"log_debug.txt")
    split_size = (args.total_parquet - args.offset) // args.world_size
    parquet_list = range(args.offset + args.rank * split_size,
              args.offset + (args.rank + 1) * split_size if args.rank != args.world_size - 1 else args.total_parquet)
    parquet_list = ['data/train-' + str(element).zfill(5) + '*' for element in parquet_list]
    _logger.info(f"Downloading {parquet_list} parquet files")
    if os.path.exists(path_rank / f"data_dictionary.pkl"):
        with open(path_rank / f"data_dictionary.pkl", 'rb') as f:
            data_dictionary = pickle.load(f)
    if os.path.exists(path_rank / f"faulted_parquet.pkl"):
        with open(path_rank / f"faulted_parquet.pkl", 'rb') as f:
            faulted_parquet = pickle.load(f)
    if os.path.exists(path_rank / f"incomplete_images.txt"):
        with open(path_rank / f"incomplete_images.txt", 'r') as f:
            incomplete_images = f.read().split("\n")
    if os.path.exists(path_rank / f"to_download_list.pkl"):
        with open(path_rank / f"to_download_list.pkl", 'rb') as f:
            parquet_list = pickle.load(f)
        _logger.info(f"Resuming the download with the parquet list {parquet_list}")
    if os.path.exists(path_rank / f"resume.txt"):
        with open(path_rank / f"resume.txt", 'r') as f:
            idx_directory, number_of_samples = f.read().split("\n")
            idx_directory, number_of_samples = int(idx_directory), int(number_of_samples)
    path_directory = path_rank / f"{idx_directory}"

    for parquet_idx, parquet in enumerate(parquet_list):
        _logger.info(f"Downloading parquet {parquet_idx} of {len(parquet_list)}")
        try:
            fake_dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir, split="train", data_files=parquet, verification_mode=datasets.VerificationMode.NO_CHECKS, streaming=args.streaming_parquet)
        except ConnectionError:
            _logger.info(f"Connection error, saving the parquet {parquet_idx} in the faulted_parquet list")
            faulted_parquet.append(parquet)
            with open(path_rank / f"faulted_parquet.pkl", 'wb') as f:
                pickle.dump(faulted_parquet, f)
            continue
        # Resume the the download if some images are missing
        # TODO: Test resume
        for iter_images, element in enumerate(fake_dataset):
            if iter_images % args.images_per_folder == 0:
                if not os.path.exists(path_rank / f"{idx_directory}"):
                    os.makedirs(path_rank / f"{idx_directory}")
                path_directory = path_rank / f"{idx_directory}"
                idx_directory += 1
            for i in range(4):
                if element[f'image_gen{i}']:
                    if args.save_images:
                        extension = element[f'filepath_gen{i}'].split('.')[-1]
                        path_image = path_directory / f"{element['id']}_gen{i}.{extension}"
                        element[f'image_gen{i}'].save(path_image)
                        data_dictionary[element['id']].append(path_image)
                else:
                    _logger.info(f"Image {element['id']} gen{i} not found")
                    incomplete_images.append(element['id'])
            url_path = path_directory / f"{element['id']}_url.txt"
            prompt_path = path_directory / f"{element['id']}_prompt.txt"
            data_dictionary[element['id']].append(url_path)
            data_dictionary[element['id']].append(prompt_path)
            if args.save_images:
                with open(url_path, 'w') as f:
                    f.write(element['url'])
                with open(prompt_path, 'w') as f:
                    f.write(element['original_prompt'])
            number_of_samples += 1
        with open(path_rank / f"data_dictionary.pkl", 'wb') as f:
            pickle.dump(data_dictionary, f)
        with open(path_rank / f"incomplete_images.txt", 'w') as f:
            f.write("\n".join(incomplete_images))
        with open(path_rank / f"resume.txt", 'w') as f:
            f.write(str(idx_directory) + "\n" + str(number_of_samples))
        with open(path_rank / f"to_download_list.pkl", 'wb') as f:
            pickle.dump(parquet_list[parquet_idx+1:], f)

    # saving both dictionary and uncompleted images
    with open(path_rank / f"data_dictionary.pkl", 'wb') as f:
        pickle.dump(data_dictionary, f)
    with open(path_rank / f"incomplete_images.txt", 'w') as f:
        f.write("\n".join(incomplete_images))
    with open(path_rank / f"finish_job.txt", 'w') as f:
        f.write("finish")

def load_all_dictionary(args):
    '''
    This function load all the dictionary from the different ranks and merge them
    '''
    data_dictionary = defaultdict(list)
    for i in range(args.world_size):
        path_rank = args.data_dir / f"data_d3/job-{i+args.offest_rank_folder}"
        with open(path_rank / f"data_dictionary.pkl", 'rb') as f:
            data_dictionary_rank = pickle.load(f)
        for key in data_dictionary_rank.keys():
            data_dictionary[key] += data_dictionary_rank[key]
    #with open(args.real_dictionary, 'rb') as f:
    #    real_dictionary = pickle.load(f)
    #for key in data_dictionary.keys():
    #    if real_dictionary[key] == []:
    #        _logger.info(f"Image {key} not found in the real dataset")
            # TODO: do something here? or during dataset creation?
    #    data_dictionary[key] += real_dictionary[key]
    with open(args.data_dir / f"data_dictionary_d3.pkl", 'wb') as f:
        pickle.dump(data_dictionary, f)
    return data_dictionary

def load_faulty_parquet(args):
    incomplete_images = []
    faulted_parquet = []
    #if os.path.exists(args.data_dir / f"incomplete_images.txt"):
    #    with open(args.data_dir / f"incomplete_images.txt", 'r') as f:
    #        incomplete_images = f.read().split("\n")
    #if os.path.exists(args.data_dir / f"faulted_parquet.pkl"):
    #    with open(args.data_dir / f"faulted_parquet.pkl", 'rb') as f:
    #        faulted_parquet = pickle.load(f)
    for i in range(args.world_size):
        path_rank = args.data_dir / f"data_d3/job-{i+args.offest_rank_folder}"
        if os.path.exists(path_rank / f"incomplete_images.txt"):
            with open(path_rank / f"incomplete_images.txt", 'r') as f:
                incomplete_images_rank = f.read().split("\n")
            incomplete_images += incomplete_images_rank
        if os.path.exists(path_rank / f"faulted_parquet.pkl"):
            with open(path_rank / f"faulted_parquet.pkl", 'rb') as f:
                faulted_parquet_rank = pickle.load(f)
            faulted_parquet += faulted_parquet_rank
    with open(args.data_dir / f"d3_incomplete_images.txt", 'w') as f:
        f.write("\n".join(incomplete_images))
    with open(args.data_dir / f"d3_faulted_parquet.pkl", 'wb') as f:
        pickle.dump(faulted_parquet, f)
    return incomplete_images, faulted_parquet

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_name', type=str, default='')
    argparser.add_argument('--world_size', type=int, default=1)
    argparser.add_argument('--offset', type=int, default=0)
    argparser.add_argument('--total_parquet', type=int, default=0)
    argparser.add_argument('--rank', type=int, default=0)
    argparser.add_argument('--offest_rank_folder', type=int, default=0)
    argparser.add_argument('--num_process', type=int, default=1)
    argparser.add_argument('--data_dir', type=Path, default='')
    argparser.add_argument('--num_files', type=Path, default='')
    argparser.add_argument('--images_per_folder', type=int, default=1000)
    argparser.add_argument('--real_dictionary', type=Path, default='/work/publicfiles/drive/elsa_dataset/laion_real_subset_3M/data_dictionary_real.pkl')
    argparser.add_argument('--download_data', action='store_true', default=False)
    argparser.add_argument('--load_data', action='store_true', default=False)
    argparser.add_argument('--load_faulty_parquet', action='store_true', default=False)
    argparser.add_argument('--save_images', action='store_true', default=False)
    argparser.add_argument('--streaming_parquet', action='store_true', default=False)
    argparser.add_argument('--cache_dir', type=Path, default='/work/horizon_ria_elsa/Elsa_datasetv2/cache_d3')
    args = argparser.parse_args()
    if args.download_data:
        data_donwload(args)
    if args.load_data:
        data_dictionary = load_all_dictionary(args)
    if args.load_faulty_parquet:
        incomplete_images, faulted_parquet = load_faulty_parquet(args)
    print("Done")
