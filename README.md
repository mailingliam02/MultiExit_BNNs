# Combining Monte Carlo dropout with Early Exit Ensembling
This github repo contains the codebase for the Imperial MSc AI Independent Project "Combining Monte Carlo dropout with Early Exit Ensembling". The full dissertation is available under MSc_Dissertation_Final.pdf.

Abstract:
`
In this report, a novel method to improve uncertainty quantification in convolutional
neural networks is tested. Monte Carlo dropout is combined with early exit ensembling,
and is shown to improve both accuracy and uncertainty quantification across
three different models and on Cifar100 and a medical chest x-ray dataset, ChestXray
14. On average, the expected calibration error was reduced by 50%, 17.7% and
16.7% for the MSDNet, VGG-19 and ResNet-18 on Cifar100 over the best tested
methods from the literature. For the chest x-ray dataset, the combination models
can match the best methods from the literature, while needing 55% fewer FLOPs.
However, while the combination approach is found to outperform alternatives, it requires
significant hyperparameter tuning to achieve optimal results which may limit
its practical applicability in some domains.
`

The primary inspiration from this code base, and the majority of the code for the MSDNet and the distillation loss is adapted from the codebase for [Distillation-Based Training for Multi-Exit Architectures](https://ieeexplore.ieee.org/document/9009834). A citation for this paper is given below:

```
@INPROCEEDINGS{9009834,  author={Phuong, Mary and Lampert, Christoph},  
booktitle={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},   
title={Distillation-Based Training for Multi-Exit Architectures},   
year={2019},  volume={},  number={},  
pages={1355-1364},  
doi={10.1109/ICCV.2019.00144}}
```

The code for the ResNet and the bidirectional distillation loss is adapted from the codebase for paper [Students are the Best Teacher: Exit-Ensemble Distillation with Multi-Exits](https://arxiv.org/abs/2104.00299). A citation for this paper is given below:

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
The structure of this README and the ChestX-ray 14 dataset loader file were adapted from https://github.com/jonahanton/SSL_medicalimaging

## How to Run
To run the repository, run the following commands from the repo base folder:

```
pip install https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchaudio-0.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
mkdir snapshots
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

