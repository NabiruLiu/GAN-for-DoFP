import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import MyDataset, Generator
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR

def test_gan(model, dataloader, device='cuda'):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        SSIM_aop = 0.0
        SSIM_dolp = 0.0
        SSIM_s0 = 0.0
        PSNR_aop = 0.0
        PSNR_dolp = 0.0
        PSNR_s0 = 0.0
        RMSE_aop = 0.0
        RMSE_dolp = 0.0
        RMSE_s0 = 0.0
        num_samples = len(dataloader)
        
        for i, (data, aop, dolp, s0) in enumerate(dataloader):
            inputs = data.to(device)
            aop_true = aop.to(device)
            dolp_true = dolp.to(device)
            s0_true = s0.to(device)
            
            aop_pred, dolp_pred, s0_pred = model(inputs)
            
            # Calculate SSIM, PSNR, MSE
            SSIM_aop += SSIM(aop_pred, aop_true, data_range=1.0).item()
            SSIM_dolp += SSIM(dolp_pred, dolp_true, data_range=1.0).item()
            SSIM_s0 += SSIM(s0_pred, s0_true, data_range=1.0).item()

            PSNR_aop += PSNR(aop_pred, aop_true, data_range=1.0).item()
            PSNR_dolp += PSNR(dolp_pred, dolp_true, data_range=1.0).item()
            PSNR_s0 += PSNR(s0_pred, s0_true, data_range=1.0).item()
            
            RMSE_s0 += torch.mean((s0_pred - s0_true) ** 2).item()
            RMSE_aop += torch.mean((aop_pred - aop_true) ** 2).item()
            RMSE_dolp += torch.mean((dolp_pred - dolp_true) ** 2).item()
            
        SSIM_aop /= num_samples
        SSIM_dolp /= num_samples
        SSIM_s0 /= num_samples
        PSNR_aop /= num_samples
        PSNR_dolp /= num_samples
        PSNR_s0 /= num_samples
        RMSE_aop /= num_samples
        RMSE_dolp /= num_samples
        RMSE_s0 /= num_samples
        
        print(f'SSIM of AoP: {SSIM_aop}')
        print(f'SSIM of DoLP: {SSIM_dolp}')
        print(f'SSIM of S0: {SSIM_s0}\n')

        print(f'RMSE of AoP: {RMSE_aop}')
        print(f'RMSE of DoLP: {RMSE_dolp}')
        print(f'RMSE of S0: {RMSE_s0}\n')

        print(f'PSNR of AoP: {PSNR_aop}')
        print(f'PSNR of DoLP: {PSNR_dolp}')
        print(f'PSNR of S0: {PSNR_s0}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_file_path = 'data/your_test_dataset.h5'
    val_dataset = MyDataset(file_path=test_file_path, transform=None)
    val_loader = DataLoader(val_dataset, batch_size=3, num_workers=2, pin_memory=True,  shuffle=False)
    model = Generator()
    checkpoint_path = 'ckpy/your_checkpoint.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model = model.to(device)
    test_gan(model=model, dataloader=val_loader)
