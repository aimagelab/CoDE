import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn import manifold
import time
import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import wandb
import torch.distributed as dist

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger = logging.getLogger('train')

DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_BATCHES = 10

__all__ = ['plot_embedding', 'get_features']

@torch.no_grad()
def get_features(dataloader, model, args):
    model.eval()
    all_features = []
    all_labels = []
    keys = []
    with torch.no_grad():
        for idx, elements in enumerate(dataloader):
            #print(f'iter: {idx}')
            if DEBUG and idx == DEBUG_BATCHES:
                break
            if not "gan" in args.dataset:
                key, real, fake, augmented_real, augmented_fake, dino_image_real_global, dino_image_real_local, dino_image_fake_global, dino_image_fake_local, fake_0, fake_1, fake_2, fake_3 = elements
                for element in key:
                    keys.append(element)
                augmented_real = augmented_real.to(device)
                # TODO: Check dataset but it seems that for single / fixed generator onlu augmented fake is returned
                if fake_0[0] is not None:
                    fake_0, fake_1, fake_2, fake_3 = fake_0.to(device), fake_1.to(device), fake_2.to(device), fake_3.to(
                        device)
                    augmented_fake = torch.cat([fake_0, fake_1, fake_2, fake_3], dim=0)
                    fake_0_labels = torch.ones(fake_0.shape[0], dtype=torch.long).to(device)
                    fake_1_labels = torch.full((fake_1.shape[0],), 2, dtype=torch.long).to(device)
                    fake_2_labels = torch.full((fake_2.shape[0],), 3, dtype=torch.long).to(device)
                    fake_3_labels = torch.full((fake_3.shape[0],), 4, dtype=torch.long).to(device)
                    fake_labels = torch.cat([fake_0_labels, fake_1_labels, fake_2_labels, fake_3_labels], dim=0)
                else:
                    augmented_fake = augmented_fake.to(device)
                    fake_labels = torch.ones(augmented_fake.shape[0], dtype=torch.long).to(device)

                real_labels = torch.zeros(augmented_real.shape[0], dtype=torch.long).to(device)
                labels = torch.cat([real_labels, fake_labels], dim=0)
                input = torch.cat([augmented_real, augmented_fake], dim=0)

            else:
                torch_list = [torch.as_tensor(elements[0][i]) for i in range(len(elements[0]))]
                input = torch.cat(torch_list, dim=0).to(device)

                lab = [torch.full((torch.as_tensor(elements[0][0]).shape[0],), i, dtype=torch.long).to(device) for i in
                       range(len(elements[0]))]
                labels = torch.cat(lab, dim=0)

            output = model(input)

            all_features.append(output.reshape(-1, output.shape[-1]).detach().cpu())
            all_labels.append(labels.reshape(-1).detach().cpu())
            # print("Batch", idx, "of", len(dataloader))
    # add time to file_name to avoid overwriting
    # now = datetime.now()
    with open(f'{time.strftime("%Y%m%d-%H%M%S")}_keys_rank_{dist.get_rank()}.txt', 'w') as f:
        for item in keys:
            f.write("%s\n" % item)
    all_features = torch.cat(all_features).to(device)
    all_labels = torch.cat(all_labels).to(device)
    return all_features, all_labels


def plot_embedding(X, Y):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    colors = [
        'xkcd:orangered',
        # 'xkcd:magenta',
        'xkcd:azure',
        # 'xkcd:green',
    ]

    plt.figure()
    for dataset_i in range(len(colors)):
        this_X = X[Y == dataset_i, :]
        plt.scatter(this_X[:, 0], this_X[:, 1], s=4, c=colors[dataset_i])
    plt.legend(["Real", "Fake"], loc=2, prop={'size': 13})
    plt.xticks([]), plt.yticks([])


def plot_embedding_fancy(X, Y, args):
    markers_setting = ['o', '^', 'v', 'p', 'H']
    colors = ['#eb4d4b', '#22a6b3', '#FFD347', '#160B09', '#B0E581']
    # colors = ['#eb4d4b', '#22a6b3', '#FFD347', '#DF7162', '#B0E581']

    sns.set_context('poster')
    sns.set(rc={'axes.facecolor': 'whitesmoke', 'axes.grid': False,
                'figure.figsize': (10, 7), 'figure.dpi': 600,
                }, font_scale=1.5
            )  # 'font.family': 'sans-serif', 'font.sans-serif': 'Calibri'
    # normalize data X
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    labels = []
    for l in Y:
        if l == 0:
            labels.append("Real")
        elif l == 1:
            labels.append("Fake_0")
        elif l == 2:
            labels.append("Fake_1")
        elif l == 3:
            labels.append("Fake_2")
        elif l == 4:
            labels.append("Fake_3")
        else:
            labels.append("Fake")

    # data frame with 3 columns: x, y, labels
    df = pd.DataFrame(X, columns=['x', 'y'])
    df['labels'] = labels
    df = df.sort_values(by=['labels'])

    fig, ax = plt.subplots(1, 1)
    # create scatter plot with seaborn of df
    sns.scatterplot(data=df, x='x', y='y', hue="labels", palette=colors, s=15, legend=True)

    # add legend on top of plot
    # plt.legend(labels=["Real", "Fake"], loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, markerscale=1.5)
    ax.grid(color='gainsboro', linestyle='--', linewidth=1, axis='both')
    # ax.get_legend().remove()

    # h, l = ax.get_legend_handles_labels()
    # legend = plt.figlegend(
    #     handles=h[0:8:1],
    #     labels=l[0:8:1],
    #     facecolor='white',
    #     ncol=2,
    #     loc='upper center',
    #     markerscale=3,
    #     # mode='expand',
    #     # bbox_to_anchor=(0.5, 0., 0.5, 0.5)
    # )

    # remove xlabel and ylabel
    plt.title(f'T-SNE with {args.model}')
    plt.xlabel('')
    plt.ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # if not args.save_model:
    #    plt.show()


def visualization(dataloader, model, output_dir, epoch, args, transf=False, generator=None, distributed=True,
                  test_features=None, test_labels=None):
    # ----------------- Training -----------------
    if test_features is None:
        test_features, test_labels = get_features(dataloader, model, args)
        if distributed:
            world_size = dist.get_world_size()
            local_rank = dist.get_rank()
            dist.barrier()
            gather_list_features = [torch.zeros_like(test_features) for _ in range(world_size)]
            dist.all_gather(gather_list_features, test_features)
            gather_list_labels = [torch.zeros_like(test_labels) for _ in range(world_size)]
            dist.all_gather(gather_list_labels, test_labels)
        else:
            gather_list_features = [test_features]
            gather_list_labels = [test_labels]

    if args.local_rank == 0:  # do this on global rank not local rank
        if test_features is None:
            concatenated_features = torch.cat(gather_list_features, dim=0)
            concatenated_labels = torch.cat(gather_list_labels, dim=0)
            print(concatenated_labels.shape)

            X = concatenated_features.cpu().numpy()
            Y = concatenated_labels.cpu().numpy()
        else:
            X = test_features
            Y = test_labels
        print("Computing t-SNE embedding")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=args.seed)
        X_tsne = tsne.fit_transform(X)

        plot_embedding_fancy(X_tsne, Y, args)

        save_path = Path(output_dir)
        image_extension = 'pdf'
        image_name = f'tsne_real_fake_val_{"transf" if transf else ""}_{args.dataset}_{generator if type(generator) == int else ""}_{epoch}.{image_extension}'

        if not Path(save_path).exists():  # create folder if not exists
            os.mkdir(save_path)

        save_path_pdf = os.path.join(save_path, image_name)
        print(f"Creating figure: {save_path_pdf} with name {image_name}")
        plt.savefig(
            save_path_pdf,
            bbox_inches='tight',
            pad_inches=0
        )

        if args.log_wandb:
            image_extension_wandb = 'jpg'
            image_name_jpg = f'tsne_real_fake_val_{"transf" if transf else ""}_{args.dataset}_{generator if type(generator) == int else ""}_{epoch}.{image_extension_wandb}'
            save_path_jpg = os.path.join(save_path, image_name_jpg)
            plt.savefig(save_path_jpg, bbox_inches='tight', pad_inches=0)
            if transf:
                wandb.log({"TSNE_plot_transf": wandb.Image(save_path_jpg)}, commit=False)
            else:
                wandb.log({"TSNE_plot": wandb.Image(save_path_jpg)}, commit=False)
            # after the log we can delete the jpg image and maintain only the pdf
            os.remove(save_path_jpg)

        print(f"Image saved at: {save_path}")
