# TumorDetection

Python Module with the implementation of a Semantic Segmentation Network for tumor detection in 
breast ultrasound images with an [EFSNet](models/efsnet.py) architecture.


## Structure

[core](core): Scripts with the final implementation for easy uses. *❗Not yet*  
[data](data): Scripts for data manipulation like loading and creating dataloaders for using the model.  
[models](models): Implementation of [EFSNet](models/efsnet.py) and its layers as well as [utilites](models/utils) 
for use.  
[utils](utils): Configurations and miscelanea.


## Motivation

Obtaining a *easy-to-use* cost-efficient deep learning model that can be used in a conventional computer and/or
in real-time.

> **Real-Time**
> It depends on final results. May it runs both in real time AND in a conventional computer or only one 
> case is plausible.


## Approach

The approach actually is a clasifier + binary segmentation model due to the image dataset.

The classification learns `normal, benign, malignant` images from the encoder part of the [model](models/efsnet.py)
while the binary segmentation model performs a segmentation of any kind of tumor wheter is beignant or not. This 
combination of predictions covers all cases due to there is no image in the dataset that has both benign and malignant 
tumors.

This way of training the model could help with data imbalance but enlarges the size of the model.


## References

**Dataset of Breast Ultrasound Images**
```
@article{al-dhabyaniDatasetBreastUltrasound2020,
  title = {Dataset of Breast Ultrasound Images},
  author = {Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
  date = {2020-02},
  journaltitle = {Data in Brief},
  shortjournal = {Data Brief},
  volume = {28},
  eprint = {31867417},
  eprinttype = {pmid},
  pages = {104863},
  issn = {2352-3409},
  doi = {10.1016/j.dib.2019.104863},
  abstract = {Breast cancer is one of the most common causes of death among women worldwide. Early detection helps in reducing the number of early deaths. The data presented in this article reviews the~medical images of breast cancer using ultrasound scan. Breast Ultrasound Dataset is categorized into three classes: normal, benign, and malignant images. Breast ultrasound images can produce great results in classification, detection, and segmentation of breast cancer when combined with machine learning.},
  langid = {english},
  pmcid = {PMC6906728},
  keywords = {Breast cancer,Classification,Dataset,Deep learning,Detection,Medical images,Segmentation,Ultrasound}
}
```


**Efficient Fast Semantic Segmentation Usin Continuous Shuffle Dilated Convolutions.**
```
@article{9063469,
  author={Hu, Xuegang and Wang, Haibo},
  journal={IEEE Access}, 
  title={Efficient Fast Semantic Segmentation Using Continuous Shuffle Dilated Convolutions}, 
  year={2020},
  volume={8},
  number={},
  pages={70913-70924},
  keywords={Semantics;Image segmentation;Feature extraction;Decoding;Real-time systems;Image resolution;Performance evaluation;Resource-constrained;semantic segmentation;continuous shuffle dilated convolution;real-time},
  doi={10.1109/ACCESS.2020.2987080}}
```


