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
2. [Citation](#citation)

## Training Dataset 
[Dataset web page](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=57) |
[🤗 HuggingFace Dataset](https://huggingface.co/datasets/elsaEU/ELSA_D3)

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
