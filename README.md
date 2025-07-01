<div align="center">
  <h1>CoDE (Contrastive Deepfake Embeddings)</h1>
  <h2>Contrasting Deepfakes Diffusion via Contrastive Learning and Global-Local Similarities

  (ECCV 2024)
  </h2>
   
</div>

<br></br>
<p align="center">
  <img src="images/deepfake_model.jpg" alt="CoDE" width="820" />

</p> 

## Table of Contents

1. [Training Dataset](#training-dataset)
2. [Dataset Creation](#dataset-creation)
3. [Training Code](#training-code)
4. [Inference](#inference)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Training Dataset 
[🎯 Project web page](https://aimagelab.github.io/CoDE/) |
[Paper](https://arxiv.org/pdf/2407.20337) |
[Dataset web page](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=57) |
[D<sup>3</sup> Test Set](https://ailb-web.ing.unimore.it/publicfiles/drive/elsa_dataset/d3_test.tar) |
[🤗 HuggingFace Dataset](https://huggingface.co/datasets/elsaEU/ELSA_D3) |
[🤗 HuggingFace Model](https://huggingface.co/aimagelab/CoDE) |

The Diffusion-generated Deepfake Detection (D<sup>3</sup>) Dataset is a comprehensive collection designed for large-scale deepfake detection. It includes 9.2 million generated images, created using four state-of-the-art diffusion model generators. Each image is generated based on realistic textual descriptions from the LAION-400M dataset.

- **Images**: 11.5 million images
- **Records**: 2.3 million records
- **Generators**: Stable Diffusion 1.4, Stable Diffusion 2.1, Stable Diffusion XL, and DeepFloyd IF
- **Aspect Ratios**: 256x256, 512x512, 640x480, 640x360
- **Encodings**: BMP, GIF, JPEG, TIFF, PNG

The D<sup>3</sup> dataset is part of the European Lighthouse on Secure and Safe AI (ELSA) project, which aims to develop effective solutions for detecting and mitigating the spread of deepfake images in multimedia content.

To try D<sup>3</sup> you can access it using

```python
from datasets import load_dataset
elsa_data = load_dataset("elsaEU/ELSA_D3", split="train", streaming=True)
```
The test set of D<sup>3</sup> is available at this link [D<sup>3</sup> Test Set](https://ailb-web.ing.unimore.it/publicfiles/drive/elsa_dataset/d3_test.tar) 

## Dataset Creation

We use [WebDataset](https://github.com/webdataset/webdataset) for efficient dataset storage and loading. WebDataset allows us to store large-scale image datasets as collections of tar files (shards), which can be streamed directly during training. This approach is particularly well-suited for distributed and high-throughput training scenarios, as it minimizes I/O bottlenecks and supports flexible data pipelines. To create your own dataset in WebDataset format, refer to the scripts in the `data_download` or `dataset/shards` directories, or consult the [WebDataset documentation](https://github.com/webdataset/webdataset).

An example of how to construct the dataset can be found in `src/debug/create_pkl.py`, while `src/data_download/wds_creation.py` is used to generate the WebDataset used during training."

## Training Code

The training process is launched using the provided shell script:

```bash
cd src
bash scripts/train.sh
```

This script launches distributed training using PyTorch's `torch.distributed.run` module and executes `train.py` with a set of recommended arguments. By default, it uses the configuration file at `config/config-training.yaml` and expects the dataset in WebDataset format (see the section above).

You can customize training by editing `scripts/train.sh` or by passing additional arguments to `train.py`. Key arguments include:
- `--config`: Path to the YAML configuration file.
- `--dataset`: Name of the dataset to use (e.g., `elsa_v2_train_dino`).
- `--model`: Model architecture (e.g., `vit_tiny_patch16_224`).
- `--epochs`: Number of training epochs.
- `--batch-size`: Batch size per process.
- `--output`: Output directory for logs and checkpoints.


## Inference
To set up the environment, please use the provided requirements file. Note that the configuration has been tested with `python 3.8.16`.
```bash
pip install -r requirements.txt
```

After downloading the test set of D<sup>3</sup> at the previous link, you can use the following code to load the dataset and run the inference on the CoDE model.

Substitute the path of the directories in ```src/inference/dataset_paths_d3.py```

```python
cd src/inference
python validate_d3.py --classificator_type "linear"
# options for classificator_type are ["linear", "knn", "svm"]
```
## Citation

Please cite with the following BibTeX:
```
@inproceedings{baraldi2024contrastive,
  title={{Contrasting Deepfakes Diffusion via Contrastive Learning and Global-Local Similarities}},
  author={Baraldi, Lorenzo and Cocchi, Federico and Cornia, Marcella and Baraldi, Lorenzo and Nicolosi, Alessandro and Cucchiara, Rita},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2024}
}
```

## Acknowledgements
We acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources and support. This work has
been supported by the Horizon Europe project "European Lighthouse on Safe and Secure AI (ELSA)" (HORIZON-CL4-2021-HUMAN-01-03), 
co-funded by the European Union.

<img src="images/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="images/FundedbytheEU.png" alt="europe" style="width:240px;"/>

