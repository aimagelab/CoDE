import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import pickle
from collections import defaultdict
import argparse
from PIL import Image

#TODO: use job array and activarte the relative function
def create_dict_real(args):
    data_dictionary = defaultdict(list)
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if not file.endswith(".txt") and not file.endswith(".json") and not file.endswith(".pkl"):
                path_image = Path(root) / file
                id = file.split("_")[0]
                # try:
                #     Image.open(path_image).convert('RGB')
                # except:
                #     print(f"Image {path_image} not found")
                #     continue
                data_dictionary[id].append(path_image)
    with open(args.data_dir / Path(f"data_dictionary_real.pkl"), 'wb') as f:
        pickle.dump(data_dictionary, f)

def create_log_real(args):
    data_dictionary = defaultdict(list)
    for root, dirs, files in os.walk(os.path.join(args.data_dir, f"part-0{str(args.rank).zfill(2)}")):
        for file in files:
            if not file.endswith(".txt") and not file.endswith(".json") and not file.endswith(".pkl"):
                path_image = Path(root) / file
                id = file.split("_")[0]
                try:
                    Image.open(path_image).convert('RGB')
                except:
                    data_dictionary[id].append(path_image)

    with open(os.path.join(args.data_dir, 'log_failed', f"failed_dict_log_real_{str(args.rank).zfill(2)}.pkl"), 'wb') as f:
        pickle.dump(data_dictionary, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args()
    #create_dict_real(args)
    create_log_real(args)
    print("done")