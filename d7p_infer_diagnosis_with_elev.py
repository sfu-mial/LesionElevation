import os
import random

import numpy as np
import torch
from comet_ml import Experiment
from pycm import ConfusionMatrix
from torchvision import transforms

from data import derm7point_diag_dataset

################################################################################

experiment = Experiment(project_name="PROJECT_NAME")

# TODO: Specify the modality.
# "derm" or "clinic"
MODALITY = None

# TODO: Adjust the path to the directory containing the model weights as
# needed.
MODEL_WEIGHTS_DIR = "./saved_models/"
# The paths below are for the models trained on the derm7pt dataset with
# ground truth elevation labels.
# If you want to use/evaluate models trained on the other 5 datasets and their
# corresponding "inferred" elevation labels, those are available in the
# `saved_models/DiagPredModels_InferredElevation/` directory.
# You will need to use the models that are named in the following format:
# - `{DatasetName}_S.pth`: For models trained on the probabilistic "soft"
#   inferred elevation labels.
# - `{DatasetName}_D.pth`: For models trained on the one-hot encoded
#   "discrete" inferred elevation labels.
# See the README.md file for more details.
MODEL_WEIGHTS = {
    "derm": MODEL_WEIGHTS_DIR
    + "DiagPredModels_GTElevation/Dermoscopic/GTDepth_model.pth",
    "clinic": MODEL_WEIGHTS_DIR
    + "DiagPredModels_GTElevation/Clinical/GTDepth_model.pth",
}

# TODO: Add the path to the directory containing the images.
IMG_DIR = None

# TODO: Adjust the path to the directory containing the file lists as needed.
# "./derm7pt_dataset_splits/Clinical/" or "./derm7pt_dataset_splits/Dermoscopic/" depending on the modality.
FILE_LIST_DIR = {
    "derm": "./derm7pt_dataset_splits/Dermoscopic/",
    "clinic": "./derm7pt_dataset_splits/Clinical/",
}

# At the moment, the code expects that this list contains 3 partitions:
# - `train`
# - `valid`
# - `test`
# However, since this is inference code, we only need to test on the `test` partition.
SPLITS = ["test"]
################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

seed_value = 8888
os.environ["PYTHONHASHSEED"] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

num_classes = 5
num_epochs = 50

# Diagnosis class label to integer mapping.
# 0 is `BCC`.
# 1 is `NEV`.
# 2 is `MEL`.
# 3 is `MISC`.
# 4 is `SK`.
class_labels = ["BCC", "NEV", "MEL", "MISC", "SK"]

# This should load a `VGG16_Depth` model as defined in `networks.py`.
model = torch.load(MODEL_WEIGHTS[MODALITY])
model.to(device)
model.eval()

test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dsets, dloaders = {}, {}

for split in SPLITS:
    dsets[split] = derm7point_diag_dataset(
        img_dir=IMG_DIR,
        idx_dir=FILE_LIST_DIR[MODALITY],
        data_split=split,
        modality=MODALITY,
        transform=test_transforms,
    )

    dloaders[split] = torch.utils.data.DataLoader(
        # Note that for testing, we set the batch size to 1 and disable
        # shuffling.
        # However, if we were to use this code for training/validation, we
        # would need to set the batch size to a larger value, and enable
        # shuffling on the training split.
        dsets[split],
        batch_size=1,
        shuffle=False,
    )

dataset_sizes = {split: len(dsets[split]) for split in SPLITS}

print(f"Dataset sizes: {dataset_sizes}")


with experiment.test():
    lbls_test, prds_test = None, None

    for inputs, depths, labels in dloaders["test"]:
        # Remember that the `depths` tensor is a 1D tensor of shape
        # `(batch_size,)`.
        # We need to convert it to a 2D tensor of shape `(batch_size, 3)`, where
        # each row is a one-hot encoded vector representing the depth label.
        # For example, if `depths` is `torch.tensor([0, 2, 1, 2])`, the resulting
        # tensor will be `torch.tensor([[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1]])`,
        # where each row is the one-hot encoded vector for the corresponding depth
        # label.
        depths = (
            torch.FloatTensor(depths.shape[0], 3)
            .zero_()
            .scatter_(1, depths.unsqueeze(1), 1)
        )

        inputs, depths, labels = (
            inputs.to(device),
            depths.to(device),
            labels.to(device),
        )

        with torch.set_grad_enabled(False):
            # Forward pass to compute the probabilistic diagnosis predictions.
            outputs = model(inputs, depths)

            # Get the predicted class label.
            _, preds = torch.max(outputs, 1)

            # And then you can do whatever you want with the predicted probabilities and/or the predicted class label.
            if lbls_test is None and prds_test is None:
                lbls_test, prds_test = labels, preds
            else:
                lbls_test = torch.cat([lbls_test, labels])
                prds_test = torch.cat([prds_test, preds])

    lbls_test, prds_test = lbls_test.cpu(), prds_test.cpu()

    experiment.log_confusion_matrix(
        y_true=lbls_test,
        y_predicted=prds_test,
        labels=class_labels,
    )

    test_cm = ConfusionMatrix(
        actual_vector=lbls_test.numpy(),
        predict_vector=prds_test.numpy(),
        digit=5,
    )

    experiment.log_metrics(
        {
            "acc": test_cm.ACC_Macro,
            "prec": test_cm.PPV_Macro,
            "rec": test_cm.TPR_Macro,
            "f1": test_cm.F1_Macro,
            "auroc": np.mean(
                np.asarray([v for (k, v) in test_cm.AUC.items()])
            ),
            "bal_acc": np.mean(
                np.asarray([v for (k, v) in test_cm.TPR.items()])
            ),
        },
    )
