import webdataset as wds
import torch
import sys
import numpy as np
import random
from local_utils.data_utils import collate_fn
from functools import partial
from tqdm import tqdm

from dataset.DeepFakeDataset import DeepFakeDataset
def create_loader_webdataset(args, shard, dataset_path, transform, batch_size, workers, double_contrastive, generator=None, data_len=None, process_type="cocofake", shuffle=False, use_transform=True, distributed=True, resampling=False, training_step=None):
    shards = open(shard).readlines()
    shards = [str(str(dataset_path) + "/" + s.strip()) for s in shards]
    # train dataset
    dataset = DeepFakeDataset(args, shards, transform, double_contrastive=double_contrastive, generator=generator, shuffle=shuffle, batch_size=batch_size, process_type=process_type, use_transform=use_transform, resampling=resampling).dataset
    if distributed and not resampling and data_len:
        dataset = dataset.with_epoch(data_len//(batch_size*torch.distributed.get_world_size()*workers))
    #else: dataset = dataset.with_epoch(data_len//(batch_size*workers))
    dataloader = wds.WebLoader(dataset, batch_size=None, num_workers=workers, pin_memory=False, collate_fn=None)
    if training_step:
        dataloader = dataloader.with_length(training_step)
    elif data_len:
         if distributed:
            dataloader = dataloader.with_length(data_len //(batch_size*torch.distributed.get_world_size()*workers) * workers)
    #else: dataloader = dataloader.with_length(data_len //(batch_size*workers) * workers) if data_len else dataloader
    return dataloader, dataset

