# Combining Monte Carlo dropout with Early Exit Ensembling
This github repo contains the codebase for the Imperial MSc AI Independent Project "Combining Monte Carlo dropout with Early Exit Ensembling".

The primary inspiration from this code base, and the majority of the code for the MSDNet and the distillation loss is adapted from [Distillation-Based Training for Multi-Exit Architectures](https://ieeexplore.ieee.org/document/9009834)

```
@INPROCEEDINGS{9009834,  author={Phuong, Mary and Lampert, Christoph},  
booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},   
title={Distillation-Based Training for Multi-Exit Architectures},   
year={2019},  volume={},  number={},  
pages={1355-1364},  
doi={10.1109/ICCV.2019.00144}}
```

The code for the ResNet and the bidirectional distillation loss is adapted from the codebase for paper [Students are the Best Teacher: Exit-Ensemble Distillation with Multi-Exits](https://arxiv.org/abs/2104.00299):

```
@misc{https://doi.org/10.48550/arxiv.2104.00299,
  doi = {10.48550/ARXIV.2104.00299},
  url = {https://arxiv.org/abs/2104.00299},
  author = {Lee, Hojung and Lee, Jong-Seok},
  title = {Students are the Best Teacher: Exit-Ensemble Distillation with Multi-Exits},
  publisher = {arXiv},
  year = {2021},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## How to Run
To run the repository, run the following commands from the repo base folder:

```
pip install requirements.txt
mkdir snapshots
mkdir runs_db
mkdir snapshots/figures
mkdir models/model_weights
```

Make sure that the relevant datasets are downloaded and stored in the data file. See the datasets section for how to set them up.
If using the pretrained models, download the weights from: https://download.pytorch.org/models/vgg19_bn-c79401a0.pth and place the .pth file under models/model_weights.

To train and evaluate an MSDNet with dropout layers (p=0.125) in the exits and after every block on Cifar100:
```
python3 main.py --full_analysis_and_save True --backbone msdnet --grad_clipping 0 --n_epochs 300 --dropout_exit True --dropout_p 0.125 --dropout_type block --reducelr_on_plateau True
```

To train and evaluate a ResNet with dropout layers (p=0.5) in the exits and after every layer on Cifar100:
```
python3 main.py --full_analysis_and_save True --backbone resnet18 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.5 --dropout_type layer --reducelr_on_plateau True
```

To train and evaluate a VGG19 with dropout layers (p=0.125) in the exits only on Cifar100 with grad clipping of 2:
```
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 2 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --reducelr_on_plateau True
```

To train and evaluate a VGG19 with dropout layers (p=0.125) in the exits and after every block with gradient accumulation on ChestX-ray 14:
```
python3 main.py --full_analysis_and_save True --backbone vgg19 --grad_clipping 0 --n_epochs 200 --dropout_exit True --dropout_p 0.125 --dropout_type block --reducelr_on_plateau True --dataset_name chestx --grad_accumulation 16
```
 
## Datasets
The data folder should be made to look like the below:

    ├── chestx
       ├── Data_Entry_2017.csv
       ├── images
    ├── cifar100
       ├── cifar-100-batches-py
         
    
The datasets can be downloaded here:
ChestX-ray 14: https://www.kaggle.com/nih-chest-xrays/data,
Cifar100: https://pytorch.org/vision/stable/datasets.html,

To get the ChestX-ray 14 dataset in the correct format, each of the image folders which are unpacked from the download (images_001, images_002,...) need to be combined into a single folder "images".

