# create file to have t-sne plots on different data types
import argparse
from pathlib import Path
import os
import wandb
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, Normalize
from torch.utils.data import DataLoader

from dataset.dataloader import create_loader_webdataset
from visualization.embedding_visualization import *
from args_train import get_args_parser, _parse_args
from timm.models import create_model
from dataset.CustomTransform import MassiveTransform

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_BATCHES = 5

def main(args):
    if args.log_wandb:
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, name=args.experiment, config=args)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=3,
        num_classes=0,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )
    model = model.to(device)

    if 'elsa' in args.dataset:
        dataset_type= ['elsa_v2_all', 'elsa_v2_one_gen']
        iter_run= [1, 4]
        transform = MassiveTransform(args)
        for i,el in enumerate(dataset_type):
            for j in range(iter_run[i]):
                args.dataset= el
                loader_eval_plot, dataset_eval_plot = create_loader_webdataset(shard=args.val_shards, dataset_path=args.data_dir, generator=j, data_len=args.data_len_eval,
                                                                               transform=transform, batch_size=args.batch_size, shuffle=False,
                                                                               process_type=el, use_transform=False, workers= args.workers_validate, distributed=False)
                visualization(loader_eval_plot, model, args.output, 1, args, generator=j, distributed=False)
    wandb.finish() if args.log_wandb else None


if __name__ == '__main__':
    print('start')
    parser, config_parser = get_args_parser()
    args, args_text = _parse_args(parser, config_parser)

    main(args)
    print('done')
