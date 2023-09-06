import argparse
import warnings
import torch
import torch.optim as optim
from dataset.noise_dataset import read_images_path, add_image_noise
from model.residual_model import Network
from train import Train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DS-N2N image denoising method')
    parser.add_argument('--image_folder', type=str, default='',
                        help='Image dataset path')
    parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
    parser.add_argument('--max_epoch', type=int, default=10000, help='max number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--step_size', type=int, default=9000, help='step size of changing the learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='factor by which learning rate decays')
    parser.add_argument('--embedding', type=int, default=48, help='image convolution embedding channels')
    # noise parameters
    parser.add_argument('--noise_type', type=str, default='guass', help='guass or poiss')
    parser.add_argument('--noise_level', type=int, default='25', help='guass: σ or poiss: λ')
    # psnr, ssim, loss data save path
    parser.add_argument('--psnr_data_save_path', type=str, default='./data/25g_psnr.txt')
    parser.add_argument('--ssim_data_save_path', type=str, default='./data/25g_ssim.txt')
    parser.add_argument('--loss_data_save_path', type=str, default='./data/25g_loss.txt')

    args = parser.parse_args()
    image_folder = args.image_folder
    device = args.device
    if device == 'cuda':
        if torch.cuda.is_available() is not True:
            warnings.warn('cuda is not available, cpu is available', UserWarning)
            device = 'cpu'
    print(f"train mode is {device} -------------------------------->>>>>>>>>>>>>>>>")
    max_epoch = args.max_epoch
    lr = args.lr
    step_size = args.step_size
    gamma = args.gamma
    image_embedding = args.embedding
    noise_type = args.noise_type
    noise_level = args.noise_level
    psnr_data_save_path = args.psnr_data_save_path
    ssim_data_save_path = args.ssim_data_save_path
    loss_data_save_path = args.loss_data_save_path

    # average psnr
    sum_psnr = []
    sum_ssim = []
    sum_loss = []
    average_psnr = 0
    average_ssim = 0
    average_loss = 0

    # read image as tensor
    images_path = read_images_path(image_folder)
    for image_path in images_path:
        print(f"image path: {image_path}")
        noise_image, clean_image = add_image_noise(image_path, noise_level, noise_type)

        Model = Network(input_channels=3, embedding=image_embedding)
        print("The number of parameters of the network is: ",
              sum(p.numel() for p in Model.parameters() if p.requires_grad))
        optimizer = optim.AdamW(Model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)

        train = Train(image_embedding, Model, noise_image, clean_image, image_path, optimizer, scheduler,
                      max_epoch, device, psnr_data_save_path, ssim_data_save_path, loss_data_save_path)
        denoised_psnr, denoised_ssim, denoised_loss = train.start()
        sum_psnr.append(denoised_psnr)
        sum_ssim.append(denoised_ssim)
        sum_loss.append(denoised_loss)

    average_psnr = sum(sum_psnr) / len(sum_psnr)
    average_ssim = sum(sum_ssim) / len(sum_ssim)
    average_loss = sum(sum_loss) / len(sum_loss)
    print(f"{image_folder}: average denoised psnr {average_psnr}, average denoised ssim {average_ssim}, average loss {average_loss}")
