import numbers

import torch
import torchvision.transforms.functional as trF


def softmax_custom(x):
    """
    Compute softmax values for each sets of scores in x.
    Source: https://stackoverflow.com/q/34968722
    """
    e_x = torch.exp(x - torch.max(x))
    return e_x / e_x.sum()


class RatioCenterCrop(object):
    """
    Center crop the image to a given ratio.
    Taken from: https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/transforms.py
    """

    def __init__(self, ratio=1.0):
        assert ratio <= 1.0 and ratio > 0
        # new_size = 0.8 * min(width, height)
        self.ratio = ratio

    def __call__(self, image):
        width, height = image.size
        new_size = self.ratio * min(width, height)
        img = trF.center_crop(image, new_size)
        return img


class CenterCrop(object):
    """
    Center crop the image to a given size.
    Taken from: https://github.com/SaoYan/IPMI2019-AttnMel/blob/master/transforms.py
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image):
        img = trF.center_crop(image, self.size)
        return img
