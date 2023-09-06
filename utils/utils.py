import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def save_to_txt(data, file_path):
    with open(file_path, 'a') as file:
        file.write(str(data) + '\n')

def PIL_To_Tensor(pil_format):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image = transform(pil_format)
    return image


def Test_Denosie_PSNR(model: nn.Module, noise_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(noise_img - model(noise_img), 0, 1)
        MSE = nn.MSELoss()(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)
    return PSNR


def Test_Origin_PSNR(origin_noise_img, origin_clean_img):
    with torch.no_grad():
        MSE = nn.MSELoss()(origin_clean_img, origin_noise_img).item()
        PSNR = 10 * np.log10(1 / MSE)
    return PSNR

def Test_Denosie_SSIM(model: nn.Module, origin_noise_img, origin_clean_img):
    with torch.no_grad():
        pred = torch.clamp(origin_noise_img - model(origin_noise_img), 0, 1)
        pred = pred.squeeze().cpu().numpy()
        clean = origin_clean_img.squeeze().cpu().numpy()

        ssim_values = []
        for i in range(pred.shape[0]):
            ssim_index = ssim(pred[i], clean[i], data_range=1.0)
            ssim_values.append(ssim_index)

        mean_ssim = np.mean(ssim_values)

        return mean_ssim


def Denoise(model, noise_img):
    with torch.no_grad():
        pred = torch.clamp(noise_img - model(noise_img), 0, 1)
    return pred


def show_clean_noise_image(clean_image, noise_image):
    img1 = clean_image.cpu().squeeze(0).permute(1, 2, 0)
    img2 = noise_image.cpu().squeeze(0).permute(1, 2, 0)

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(img1.detach().numpy())
    ax[0].set_title('Clean Img')

    ax[1].imshow(img2.detach().numpy())
    ax[1].set_title('Noise Img')

    plt.show()
