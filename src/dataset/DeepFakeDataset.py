import os
import random
from abc import abstractmethod
from pathlib import Path

import torch
import torch.utils.data as data
import webdataset as wds
import json
from torchvision import transforms
from collections import defaultdict

class ProcessData():
    def __init__(self, transform, use_transform=True):
        self.transform = transform
        self.use_transform = use_transform
        self.dictionary_return = {'key':None, 'real_image':None, 'fake':None, 'augmented_real':None, 'augmented_fake':None, 'dino_image_real_global':None,
                                  'dino_image_real_local':None, 'dino_image_fake_global':None,
                                  'dino_image_fake_local':None, 'fake_0': None, 'fake_1': None,
                                  'fake_2': None, 'fake_3': None}

    def dict_to_list(self):
        return [self.dictionary_return[key] for key in self.dictionary_return.keys()]

    @abstractmethod
    def process_data(self, item):
        '''
        This is the function that processes the data. It should return a list of content that must respect this convention:
        [key, real_image, fake, augmented_real, augmented_fake, dino_image_real_global, dino_image_real_local, dino_image_fake_global, dino_image_fake_local]
        If some of the images are not available, return None to avoid confusions.
        If an input is added, add it to the list and to the return statement.
        :param item:
        :return:
        '''
        pass

# TODO: fix hard_code in the path
class Filter_wds():
    def __init__(self, args, path= 'dataset/filter_real_images.json'):
        if args.cineca: self.path = 'dataset/filter_real_images.json'
        else: self.path = path

        with open(self.path, 'r') as f:
            data = json.load(f)
        self.data = data

    def filter_function(self, sample):
        # For example, let's keep only items where 'key' equals 'value'
        if sample['__key__'].split("__")[-1] not in self.data:
            return True
        else:
            print(f'Problematic key to be filtered: {sample["__key__"].split("__")[-1]}')
            return False


class ElsaV2_trainDino(ProcessData):
    def __init__(self, transform, double_contrastive=False, use_transform=True):
        super().__init__(transform, use_transform=use_transform)
        self.double_contrastive= double_contrastive

    def process_data(self, item):
        key = item['__key__']

        list_key = list(item.keys())
        if 'jpg' in list_key[3]: format_generated = 'jpg'
        else: format_generated = 'png'
        if 'jpg' in list_key[7]: format_real = 'jpg'
        else: format_real = 'png'

        real_image = self.transform(item[f'real.{format_real}'], transform=False)
        augmented_real = self.transform(item[f'real.{format_real}'], transform=self.use_transform)

        idx = random.sample(range(4), 1)[0]
        fake = item[f'gen_{idx}.{format_generated}']
        augmented_fake = self.transform(fake, transform=self.use_transform)

        dino_image_real_global = self.transform(item[f'real.{format_real}'], transform=self.use_transform, dino='global') # PIL image
        dino_image_real_local = self.transform(item[f'real.{format_real}'], transform=self.use_transform, dino='local')

        if self.double_contrastive:
            dino_image_fake_global = self.transform(fake , transform=self.use_transform, dino='global')  # PIL image
            dino_image_fake_local = self.transform(fake, transform=self.use_transform, dino='local')
            self.dictionary_return['dino_image_fake_global'] = dino_image_fake_global
            self.dictionary_return['dino_image_fake_local'] = dino_image_fake_local

        fake= self.transform(fake, transform=False)
        self.dictionary_return['key'] = key
        self.dictionary_return['real_image'] = real_image
        self.dictionary_return['fake'] = fake
        self.dictionary_return['augmented_real'] = augmented_real
        self.dictionary_return['augmented_fake'] = augmented_fake
        self.dictionary_return['dino_image_real_global'] = dino_image_real_global
        self.dictionary_return['dino_image_real_local'] = dino_image_real_local
        return self.dict_to_list()

# used to visulization aims
class ElsaV2_all(ProcessData):
    def __init__(self, transform, use_transform=True):
        super().__init__(transform, use_transform=use_transform)

    def process_data(self, item):
        key = item['__key__']

        list_key= list(item.keys())
        if 'jpg' in list_key[3]: format_generated= 'jpg'
        else: format_generated= 'png'
        fake_0 = self.transform(item[f'gen_0.{format_generated}'], transform=self.use_transform)
        fake_1 = self.transform(item[f'gen_1.{format_generated}'], transform=self.use_transform)
        fake_2 = self.transform(item[f'gen_2.{format_generated}'], transform=self.use_transform)
        fake_3 = self.transform(item[f'gen_3.{format_generated}'], transform=self.use_transform)

        if 'jpg' in list_key[7]: format_real= 'jpg'
        else: format_real= 'png'
        real_image = self.transform(item[f'real.{format_real}'], transform=False)
        augmented_real = self.transform(item[f'real.{format_real}'], transform=self.use_transform)
        self.dictionary_return['key'] = key
        self.dictionary_return['real_image'] = real_image
        self.dictionary_return['augmented_real'] = augmented_real
        self.dictionary_return['fake_0'] = fake_0
        self.dictionary_return['fake_1'] = fake_1
        self.dictionary_return['fake_2'] = fake_2
        self.dictionary_return['fake_3'] = fake_3

        # images = torch.cat(images, dim=0) # concat this tensors
        return self.dict_to_list()
class DeepFakeDataset():
    def __init__(self, args, urls, image_transform, batch_size, generator, double_contrastive, one_generator=False, shuffle=False, process_type="cocofake",
                 use_transform=True, resampling=False, **kwargs):
        
        if process_type == "elsa_v2_train_dino":
            process_data = ElsaV2_trainDino(image_transform, double_contrastive=double_contrastive, use_transform=use_transform).process_data
        elif process_type == "elsa_v2_all":
            process_data = ElsaV2_all(image_transform, use_transform=use_transform).process_data
        else:
            raise NotImplementedError("Process type not implemented")
        filter_method = Filter_wds(args)
        self.ds = wds.DataPipeline(
            wds.ResampledShards(urls) if resampling else wds.SimpleShardList(urls),
            wds.split_by_worker,
            wds.split_by_node,
            wds.tarfile_to_samples(),
            wds.shuffle(1000) if shuffle else None,
            wds.decode('pil'),
            wds.select(filter_method.filter_function),
            wds.map(process_data),
            wds.batched(batch_size),
            **kwargs
        )

    @property
    def dataset(self):
        return self.ds