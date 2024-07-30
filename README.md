# Lesion Elevation Prediction from Skin Images Improves Diagnosis

This is the code repository for our paper published at [Medical Image Computing and Computer-Assisted Intervention (MICCAI)](https://conferences.miccai.org/2024/en/) [ISIC Skin Image Analysis Workshop (ISIC) 2024](https://workshop.isic-archive.com/2024/):

> _Lesion Elevation Prediction from Skin Images Improves Diagnosis_<br>
> Kumar Abhishek, Ghassan Hamarneh<br>
Medical Image Analysis Lab, School of Computing Science, Simon Fraser University, Canada<br>

## Abstract

While deep learning-based computer-aided diagnosis for skin lesion image analysis is approaching dermatologists' performance levels, there are several works showing that incorporating additional features such as shape priors, texture, color constancy, and illumination further improves the lesion diagnosis performance. In this work, we look at another clinically useful feature, skin lesion elevation, and investigate the feasibility of predicting and leveraging skin lesion elevation labels. Specifically, we use a deep learning model to predict image-level lesion elevation labels from 2D skin lesion images. We test the elevation prediction accuracy on the derm7pt dataset, and use the elevation prediction model to estimate elevation labels for images from five other datasets: ISIC 2016, 2017, and 2018 Challenge datasets, MSK, and DermoFit. We evaluate cross-domain generalization by using these estimated elevation labels as auxiliary inputs to diagnosis models, and show that these improve the classification performance, with AUROC improvements of up to 6.29% and 2.69% for dermoscopic and clinical images, respectively.

## Code

--coming soon--

## Citation

If you find our work usefu, please cite our paper: 

Kumar Abhishek, Ghassan Hamarneh, "[Lesion Elevation Prediction from Skin Images Improves Diagnosis](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_isic2024b.pdf)", Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (ISIC), 2024.

The corresponding bibtex entry is:

```
@InProceedings{abhishek2024lesion,
author = {Abhishek, Kumar and Hamarneh, Ghassan},
title = {Lesion Elevation Prediction from Skin Images Improves Diagnosis},
booktitle = {Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop},
month = {October},
year = {2024}
}
```
