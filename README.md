# Lesion Elevation Prediction from Skin Images Improves Diagnosis

This is the code repository for our paper published at [Medical Image Computing and Computer-Assisted Intervention (MICCAI)](https://conferences.miccai.org/2024/en/) [ISIC Skin Image Analysis Workshop (ISIC) 2024](https://workshop.isic-archive.com/2024/):

> _Lesion Elevation Prediction from Skin Images Improves Diagnosis_<br>
> Kumar Abhishek, Ghassan Hamarneh<br>
Medical Image Analysis Lab, School of Computing Science, Simon Fraser University, Canada<br>
> [[DOI]](https://doi.org/10.1007/978-3-031-77610-6_54_14) [[PDF]](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_isic2024b.pdf) [[Oral Presentation Slides]](https://workshop.isic-archive.com/2024/slides_abhishek_b.pdf)

## Key Contributions

1. **Elevation Prediction**: First work to predict skin lesion elevation labels directly from 2D RGB images.
2. **Cross-domain Generalization**: Demonstrated that inferred elevation labels improve diagnosis on datasets without ground truth elevation.
3. **Clinical Impact**: Significant AUROC improvements for both dermoscopic and clinical images.
4. **Teledermatology Enhancement**: Potential to bridge the gap in teledermatology by providing elevation information as a proxy for in-person palpation.


## Datasets Used


### derm7pt Dataset has GT elevation information
- **Modality**: Clinical and Dermoscopic images
- **Elevation Labels**: 3 classes (`flat`, `palpable`, `nodular`)
- **Diagnosis Labels**: 5 classes (BCC, NEV, MEL, MISC, SK)
- **Metadata Format**: CSV files with image paths, elevation labels, and diagnosis labels

### Other Datasets without GT elevation information
- **ISIC 2016, 2017, 2018**: Dermoscopic images
- **MSK**: Dermoscopic images  
- **DermoFit**: Clinical images
- **Metadata Format**: CSV files with image paths and diagnosis labels (no ground truth elevation)

## Model Architecture

### Elevation Prediction Model
- **Base Architecture**: VGG-16 with Global Average Pooling (GAP)
- **Input**: 224×224 RGB images
- **Output**: 3-class elevation prediction (`flat`, `palpable`, `nodular`)

### Diagnosis Models
- **Without Elevation**: VGG-16 with Global Average Pooling (GAP)
- **With Elevation**: Same as without elevation, but with elevation labels concatenated to the output of the GAP layer
- **Input**: 224×224 RGB images + elevation labels (one-hot encoded or soft probabilities)
- **Output**: 5-class diagnosis prediction (BCC, NEV, MEL, MISC, SK)



## File Structure

```
LesionElevation/
├── networks.py              # Model architectures
├── data.py                  # Dataset classes
├── utils.py                 # Utility functions and custom transforms
├── infer_elevation.py       # Elevation prediction inference
├── d7p_infer_diagnosis_with_elev.py      # derm7pt diagnosis inference with elevation
├── d7p_infer_diagnosis_without_elev.py   # derm7pt diagnosis inference without elevation
├── other_dsets_infer_diagnosis_with_elev.py    # Other datasets' diagnosis inference with elevation
├── other_dsets_infer_diagnosis_without_elev.py # Other datasets' diagnosis inference without elevation
|── saved_models/            # Saved models (see below)
└── README.md
```

## Saved Models

The pre-trained models are hosted on 🤗 Hugging Face for easy access and reproducibility: 🤗 **[skin-lesion-elevation](https://huggingface.co/kabhishe/skin-lesion-elevation)**, which contains all the model weights organized in the same structure as expected by this codebase.

<!-- ```
[skin-lesion-elevation](https://huggingface.co/kabhishe/skin-lesion-elevation)
└── saved_models
    ├── ElevPredModels # Elevation prediction models
    │   ├── Clinical_statedict.pth # Clinical image elevation predictor
    │   └── Dermoscopic_statedict.pth # Dermoscopic image elevation predictor
    ├── DiagPredModels_GTElevation # Diagnosis Models with GT Elevation (derm7pt)
    │   ├── Clinical
    │   │   ├── GTDepth_model.pth # Clinical images with elevation
    │   │   └── NoDepth_model.pth # Clinical images without elevation
    │   └── Dermoscopic
    │       ├── GTDepth_model.pth # Dermoscopic images with elevation
    │       └── NoDepth_model.pth # Dermoscopic images without elevation
    ├── DiagPredModels_InferredElevation # Diagnosis Models with Inferred Elevation
    └── <DATASET> # Contains models for each dataset
        ├── {DATASET}_N.pth: No elevation label
        ├── {DATASET}_S.pth: Soft elevation labels
        └── {DATASET}_D.pth: Discrete elevation labels
``` -->

The [`saved_models`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models) directory has the following structure:
* [`ElevPredModels`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/ElevPredModels): Elevation prediction models
    * [`Clinical_statedict.pth`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/ElevPredModels/Clinical_statedict.pth): Clinical image elevation predictor
    * [`Dermoscopic_statedict.pth`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/ElevPredModels/Dermoscopic_statedict.pth): Dermoscopic image elevation predictor
* [`DiagPredModels_GTElevation`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/DiagPredModels_GTElevation): Diagnosis Models with GT Elevation (derm7pt)
    * [`Clinical`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/DiagPredModels_GTElevation/Clinical): Clinical images with elevation
    * [`Dermoscopic`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/DiagPredModels_GTElevation/Dermoscopic): Dermoscopic images with elevation
* [`DiagPredModels_InferredElevation`](https://huggingface.co/kabhishe/skin-lesion-elevation/tree/main/saved_models/DiagPredModels_InferredElevation): Diagnosis Models with Inferred Elevation
    * `<DATASET>`: Contains models for each dataset
        * `{DATASET}_N.pth`: No elevation label
        * `{DATASET}_S.pth`: Soft elevation labels
        * `{DATASET}_D.pth`: Discrete elevation labels


## Usage

### 1. Elevation Prediction

To predict elevation labels for images without ground truth elevation:

```python
# Configure paths in infer_elevation.py
MODEL_STATEDICT_PATH = "./saved_models/ElevPredModels/Clinical_statedict.pth"  # or Dermoscopic_statedict.pth
IMG_DIR = "/path/to/images/"
FILE_LIST_DIR = "/path/to/file_lists/"
ELEVATION_OUTPUT_DIR = "/path/to/output/"

# Run elevation inference
python infer_elevation.py
```

### 2. Diagnosis with Ground Truth Elevation (derm7pt)

```python
# Configure paths in d7p_infer_diagnosis_with_elev.py
MODALITY = "derm"  # or "clinic"
IMG_DIR = "/path/to/derm7pt/images/"

# Run diagnosis with ground truth elevation
python d7p_infer_diagnosis_with_elev.py
```

### 3. Diagnosis without Elevation (derm7pt)

```python
# Configure paths in d7p_infer_diagnosis_without_elev.py
MODALITY = "derm"  # or "clinic"
IMG_DIR = "/path/to/derm7pt/images/"

# Run diagnosis without elevation
python d7p_infer_diagnosis_without_elev.py
```

### 4. Diagnosis with Inferred Elevation (Other Datasets)

```python
# Configure paths in other_dsets_infer_diagnosis_with_elev.py
DATASET_NAME = "ISIC2017"  # DF, ISIC2016, ISIC2017, ISIC2018, MSK
ELEVATION_TYPE = "S"  # "S" for soft, "D" for discrete
ELEVATION_OUTPUT_DIR = "/path/to/inferred/elevations/"
IMG_DIR = "/path/to/images/"

# Run diagnosis with inferred elevation
python other_dsets_infer_diagnosis_with_elev.py
```

### 5. Diagnosis without Elevation (Other Datasets)

```python
# Configure paths in other_dsets_infer_diagnosis_without_elev.py
DATASET_NAME = "ISIC2017"  # DF, ISIC2016, ISIC2017, ISIC2018, MSK
ELEVATION_OUTPUT_DIR = "/path/to/inferred/elevations/"
IMG_DIR = "/path/to/images/"

# Run diagnosis without elevation
python other_dsets_infer_diagnosis_without_elev.py
```

## Abstract

<details>

<summary></summary>

While deep learning-based computer-aided diagnosis for skin lesion image analysis is approaching dermatologists' performance levels, there are several works showing that incorporating additional features such as shape priors, texture, color constancy, and illumination further improves the lesion diagnosis performance. In this work, we look at another clinically useful feature, skin lesion elevation, and investigate the feasibility of predicting and leveraging skin lesion elevation labels. Specifically, we use a deep learning model to predict image-level lesion elevation labels from 2D skin lesion images. We test the elevation prediction accuracy on the derm7pt dataset, and use the elevation prediction model to estimate elevation labels for images from five other datasets: ISIC 2016, 2017, and 2018 Challenge datasets, MSK, and DermoFit. We evaluate cross-domain generalization by using these estimated elevation labels as auxiliary inputs to diagnosis models, and show that these improve the classification performance, with AUROC improvements of up to 6.29% and 2.69% for dermoscopic and clinical images, respectively.

</details>

## Citation

If you find our work useful, please cite our paper:

Kumar Abhishek, Ghassan Hamarneh, "[Lesion Elevation Prediction from Skin Images Improves Diagnosis](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_isic2024b.pdf)", Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop (ISIC), pp. 45-55, 2024, DOI: [10.1007/978-3-031-77610-6_5](https://doi.org/10.1007/978-3-031-77610-6_5).

The corresponding bibtex entry is:

```bibtex
@InProceedings{abhishek2024lesion,
  author = {Abhishek, Kumar and Hamarneh, Ghassan},
  title = {Lesion Elevation Prediction from Skin Images Improves Diagnosis},
  booktitle = {Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) ISIC Skin Image Analysis Workshop},
  month = {January},
  volume = {15274},
  pages = {45-55},
  year = {2025},
  doi = {10.1007/978-3-031-77610-6_5},
  url = {https://link.springer.com/chapter/10.1007/978-3-031-77610-6_5}
}
```
