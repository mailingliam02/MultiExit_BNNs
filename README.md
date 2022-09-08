# Combining Monte Carlo dropout with Early Exit Ensembling
This github repo contains the codebase for all experiments for the Independent Project `Combining Monte Carlo dropout with Early Exit Ensembling` (Imperial MSc AI 2022). <br />

Abstract:
`
In this report, a novel method to improve uncertainty quantification in convolutional neural networks is tested. Monte Carlo dropout is combined with early exit ensembling, and is shown to improve both accuracy and uncertainty quantification across three different models and on Cifar100 and a medical chest x-ray dataset, ChestX-ray 14. On average, the expected calibration error was reduced by 50\%, 17.7\% and 16.7\% for the MSDNet, VGG-19 and ResNet-18 on Cifar100 over the best tested methods from the literature. For the chest x-ray dataset, the combination models can match the best methods from the literature, while needing 55\% fewer FLOPs. While the combination is found to outperform alternatives, it requires significant hyperparameter tuning to achieve optimal results. Further research is required to determine the best method for finding the best parameters. 
`
This codebase is largely inspired from the following two sources:
The code for the MSDNet and the distillation loss is adapted from [Distillation-Based Training for Multi-Exit Architectures](https://ieeexplore.ieee.org/document/9009834)

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

# How to Run
To run the repository, run the following commands from the repo base folder:

pip install requirements.txt
mkdir snapshots
mkdir runs_db
mkdir snapshots/figures

# Directory Structure

    ├── data
    ├── datasets                  
    ├── models
    ├── to_train

       
## Pretrained Models
To run the pretrained VGG-19 with preloaded model weights, download the weights from:

https://www.dropbox.com/sh/87d24jqsl6ra7t2/AADN4jKnvTI0U5oT6hTmQZz8a/PIRL.pth

And place the .pth file in X

## Datasets
The data directory should be set up with the following structure:

    ├── data
        ├── chestx
            ├── Data_Entry_2017.csv
            ├── images
        ├── CIFAR100
            ├── cifar-100-batches-py
         
    
Links for where to download each dataset are given here:
[ChestX-ray14](https://www.kaggle.com/nih-chest-xrays/data),
[CIFAR100](https://pytorch.org/vision/stable/datasets.html),

### Note:

**ChestX**: When unpacking into the chestx directory, the various image folders (images_001, images_002,...,images_012) were combined so that all image files were contained directly in a single images directory as in the above structure. This can be done by repeated usage of:
```
mv images_0XX/* images/
```

