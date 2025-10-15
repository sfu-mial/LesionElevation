import os
import random

import numpy as np
import torch
from torchvision import transforms

from data import GenericSkinLesionDataset
from networks import VGG16
from utils import softmax_custom

################################################################################

# TODO: Add the path to the appropriate model state dict `.pth` file.
# "./saved_models/ElevPredModels/Clinical_statedict.pth" or "./saved_models/ElevPredModels/Derm_statedict.pth" depending on the modality.
MODEL_STATEDICT_PATH = None

# TODO: Add the path to the directory containing the images.
IMG_DIR = None

# TODO: Change the list of data partitions if needed.
# At the moment, the code expects that this list contains 3 partitions:
# - `train`
# - `valid`
# - `test`
# Remember that since elevation labels do not exist **at all** for these datasets, we perform inference on the entire dataset, and not just on the test partition.
SPLITS = ["train", "valid", "test"]

# TODO: Add the path to the directory containing the file lists.
# At the moment, the code expects that this directory contains 3 file lists:
# - `train.csv`
# - `valid.csv`
# - `test.csv`
FILE_LIST_DIR = None

# TODO: Add the path to the directory where the inferred elevation maps will be saved.
ELEVATION_OUTPUT_DIR = None

################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

seed_value = 8888
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

num_classes = 3

# Elevation class label to integer mapping.
# 0 is `flat`.
# 1 is `palpable`.
# 2 is `nodular`.
class_labels = ["flat", "palpable", "nodular"]

model = VGG16(num_classes=num_classes, GAP=True)
model.load_state_dict(torch.load(MODEL_STATEDICT_PATH), strict=True)
model.to(device)

model.eval()

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

datasets, dataloaders = {}, {}

img_dirs, file_lists = {}, {}

for split in SPLITS:
    img_dirs[split] = IMG_DIR
    file_lists[split] = f"{FILE_LIST_DIR}/{split}.csv"

for split in SPLITS:
    datasets[split] = GenericSkinLesionDataset(
        img_dir=img_dirs[split],
        file_list=file_lists[split],
        transform=test_transforms,
    )

    dataloaders[split] = torch.utils.data.DataLoader(
        datasets[split], batch_size=1, shuffle=False
    )

dataset_sizes = {split: len(datasets[split]) for split in SPLITS}

print(f"Dataset sizes: {dataset_sizes}")


for split in SPLITS:
    with open(f"{ELEVATION_OUTPUT_DIR}/{split}.csv", "w") as file:
        for img_names, imgs, labels in dataloaders[split]:
            preds = model(imgs.to(device))
            pred_class = preds.argmax(dim=1).item()
            pred_softmax = (
                softmax_custom(preds).cpu().detach().numpy().tolist()
            )
            line = img_names[0] + ";" + str(labels.item()) + ";"
            line += (
                ",".join(map(str, pred_softmax)) + ";" + str(pred_class) + "\n"
            )
            file.write(line)

print(f"Elevation maps saved to {ELEVATION_OUTPUT_DIR}")
