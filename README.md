# DS-N2N
Dual-Sampling Noise2Noise: Efficient Single Image Denoising

## How to run

### 1. Dependences
* python == 3.8
* pytorch == 2.0.1
* skimage == 0.19.0
* tqdm == 4.50.2

### 2. Train DS-N2N
```
python main.py \
  --image_folder (your nosiy images folder, type = str) \
  --device (cuda or cpu, type = str, default = 'cuda') \
  --max_epoch (max number of epochs, type = int, default = 10000) \
  --lr (learning rate, type = float, default = 0.001) \
  --step_size (step size of changing the learning rate, type = int, default = 9000) \
  --gamma (factor by which learning rate decays, type = float, default = 0.5) \
  --embedding (image convolution embedding channels, type = int, default = 48) \
  --noise_type (guass or poiss, type = str, default = 'guass') \
  --noise_level (guass: σ or poiss: λ, type = int, default = 25) \
  --psnr_data_save_path (psnr data save path, type = str, default = './data/25g_psnr.txt') \
  --ssim_data_save_path (ssim data save path, type = str, default = './data/25g_ssim.txt') \
  --loss_data_save_path (loss data save path, type = str, default = './data/25g_loss.txt')

**NOTE**
If you want to save psnr, ssim, and loss data, please release the save file comments under train.py：
* denosied_psnr = Test_Denosie_PSNR(self.model, noise_image, clean_image)
* denosied_ssim = Test_Denosie_SSIM(self.model, noise_image, clean_image)
* progress_bar.set_postfix(loss=loss.item(), denoising_PSNR=denosied_psnr, denosied_SSIM=denosied_ssim)
* save_to_txt(denosied_psnr, self.psnr_txt_path)
* save_to_txt(denosied_ssim, self.ssim_txt_path)
* save_to_txt(loss.item(), self.loss_txt_path)
Please note that saving the data will significantly increase the denoising processing time！


## Cite
If you have referenced the content of this repository, please cite the following papers:
@article{bai2025dual,
  title={Dual-Sampling Noise2Noise: Efficient Single Image Denoising},
  author={Bai, Jibo and Zhu, Daqi and Chen, Mingzhi},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE}
}


