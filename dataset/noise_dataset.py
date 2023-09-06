import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from utils.utils import PIL_To_Tensor


def read_images_path(images_folder):
    images = os.listdir(images_folder)
    images_path = []
    for image in images:
        images_path.append(os.path.join(images_folder, image))
    return images_path


def add_image_noise(image_path, noise_level, noise_type):
    assert noise_type == "guass" or noise_type == "poiss"
    image = Image.open(image_path)
    image_tensor = PIL_To_Tensor(image)

    # add noise
    clean_image = image_tensor
    if noise_type == "guass":
        noise_image = clean_image + torch.normal(0, noise_level / 255, clean_image.shape)
        noise_image = torch.clamp(noise_image, 0, 1)

    if noise_type == "poiss":
        noise_image = torch.poisson(noise_level * clean_image) / noise_level

    return noise_image, clean_image
