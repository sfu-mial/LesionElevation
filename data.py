import random

import numpy as np
import pandas as pd
import torch
from PIL import Image


def __worker_init_fn(worker_id):
    """
    Sets the worker functions to use the same random seed for reproducibility.
    """
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)


class derm7point_diag_dataset(torch.utils.data.Dataset):
    """
    Dataset object to fetch images, depths, and diagnosis labels from the
    derm7point dataset.
    """

    def __init__(self, img_dir, idx_dir, data_split, modality, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing the images.
            idx_dir (str): Path to the directory containing the index files.
            data_split (str): The data split to use.
            modality (str): The modality to use (derm or clinic).
        """
        super(derm7point_diag_dataset, self).__init__()
        self.img_dir = img_dir
        self.idx_dir = idx_dir
        assert modality in ["derm", "clinic"]
        self.modality = modality
        self.transform = transform

        self.data = pd.read_csv("{:s}{:s}.csv".format(idx_dir, data_split))

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img, depth, diag = (
            row[self.modality],
            row["elevation_mapped"],
            row["diagnosis_mapped"],
        )

        img = Image.open(self.img_dir + img)
        if self.transform:
            img = self.transform(img)
        depth, diag = int(depth), int(diag)

        return img, depth, diag

    def __len__(self):
        return self.data.shape[0]


class GenericSkinLesionDataset(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, and diagnosis labels from
    any dataset that has file lists in CSV format with 2 items per entry:
    relative image path (including the filename extension) and the class
    label.
    """

    def __init__(self, img_dir, file_list, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing the images.
            file_list (str): Path to the CSV file containing the file list.
            transform (optional): Optional transform to be applied to the images.
        """
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform

        self.data = pd.read_csv(file_list, delimiter=",", header="infer")

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        """
        row = self.data.iloc[idx]
        img_name, label = row[0], row[1]

        img = Image.open("{:s}{:s}".format(self.img_dir, img_name))
        if self.transform:
            img = self.transform(img)
        label = int(label)

        return img_name, img, label
        # return img, label

    def __len__(self):
        return self.data.shape[0]


class GenericSkinLesionDatasetWithElevation(torch.utils.data.Dataset):
    """
    Dataset object to fetch image names, images, elevation labels, and
    diagnosis labels from datasets' CSV file lists.
    The CSV file lists have 4 columns separated by semicolons:
    - relative image path (including the filename extension).
    - diagnosis label.
    - soft elevation label, e.g, [0.1, 0.2, 0.7].
    - discrete elevation label that is the argmax of the soft elevation
      label, e.g, 2.
    """

    def __init__(self, img_dir, file_list, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing the images.
            file_list (str): Path to the CSV file containing the
                             corresponding split's file list.
            transform (optional): Optional transform to be applied to the images.
        """
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform

        # Note that in my case, I did not write headers in the CSV file lists
        # (see `infer_elevation.py`), so here I have to set `header=None`.
        self.data = pd.read_csv(file_list, delimiter=";", header=None)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to fetch.
        """
        row = self.data.iloc[idx]
        img_name, label, soft_elev, discrete_elev = (
            row[0],
            row[1],
            row[2],
            row[3],
        )

        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(f"{self.img_dir}{img_name}")
        else:
            img = Image.open(f"{self.img_dir}{img_name}.jpg")

        if self.transform:
            img = self.transform(img)

        # Convert soft elevation label to a float tensor.
        label, soft_elev, discrete_elev = (
            int(label),
            torch.FloatTensor(eval(soft_elev)),
            int(discrete_elev),
        )

        return img_name, img, label, soft_elev, discrete_elev
