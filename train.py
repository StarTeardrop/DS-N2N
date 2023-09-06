from tqdm import tqdm
from loss.customizable_loss import Custom_Loss
from utils.utils import Test_Origin_PSNR, \
                        Test_Denosie_PSNR, \
                        Test_Denosie_SSIM,\
                        Denoise, \
                        show_clean_noise_image,\
                        save_to_txt



class Train(object):
    def __init__(self, embedding, model, noise_image, clean_image, image_path, optimizer, scheduler, epoch,
                 device, psnr_txt_path, ssim_txt_path, loss_txt_path):
        super().__init__()
        self.embedding = embedding
        self.model = model.to(device)
        self.noise_image = noise_image
        self.clean_image = clean_image
        self.image_path = image_path
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = epoch
        self.psnr_txt_path = psnr_txt_path
        self.ssim_txt_path = ssim_txt_path
        self.loss_txt_path = loss_txt_path

    def start(self):
        noise_image = self.noise_image.unsqueeze(0).to(self.device)
        clean_image = self.clean_image.unsqueeze(0).to(self.device)

        custom_loss = Custom_Loss(self.model, self.device)
        origin_psnr = Test_Origin_PSNR(noise_image, clean_image)
        progress_bar = tqdm(range(self.epoch), desc=f"Origin PSNR: {origin_psnr}", leave=True)
        for _ in progress_bar:
            loss = custom_loss(noise_image)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # compute denoising PSNR
            # denosied_psnr = Test_Denosie_PSNR(self.model, noise_image, clean_image)
            # denosied_ssim = Test_Denosie_SSIM(self.model, noise_image, clean_image)
            # progress_bar.set_postfix(loss=loss.item(), denoising_PSNR=denosied_psnr, denosied_SSIM=denosied_ssim)

            # save_to_txt(denosied_psnr, self.psnr_txt_path)
            # save_to_txt(denosied_ssim, self.ssim_txt_path)
            # save_to_txt(loss.item(), self.loss_txt_path)

            # compute final PSNR
        denosied_psnr = Test_Denosie_PSNR(self.model, noise_image, clean_image)
        denosied_ssim = Test_Denosie_SSIM(self.model, noise_image, clean_image)
        return_loss = loss.item()
        print(f"image path: {self.image_path}: origin PSNR: {origin_psnr}, denoised PSNR: {denosied_psnr}, denoised SSIM: {denosied_ssim}")

        return denosied_psnr,denosied_ssim,return_loss
