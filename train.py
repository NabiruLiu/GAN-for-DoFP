import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import os
from model import MyDataset, Generator, Discriminator, CustomLoss, custom_transform, PerceptualLoss
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import GradScaler, autocast
import torchmetrics

def train(generator, discriminator, train_loader, val_loader, device, num_epochs, learning_rate, weight_decay=1e-4, checkpoint_path='ckpt/your_checkpoint.pth', savebest=True):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    scaler_G = GradScaler()
    scaler_D = GradScaler()
    
    weight_aop = 1
    weight_dolp = 1
    weight_s0 = 1

    # Loss functions
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    custom_criterion = CustomLoss().to(device)
    perceptual_criterion = PerceptualLoss().to(device)
    
    # TensorBoard
    log_dir = "logs/fit/" + time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    
    best_val_loss = float('inf')
    best_psnr = float('-inf')
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        print(f'Loaded checkpoint.')
        
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        train_loss_G = 0.0
        train_loss_D = 0.0
        start_time = time.time()
        
        for i, (data, aop, dolp, s0) in enumerate(train_loader):
            inputs = data.to(device)
            aop_true = aop.to(device)
            dolp_true = dolp.to(device)
            s0_true = s0.to(device)
            
            # Update discriminator
            optimizer_D.zero_grad()
            with autocast():
                # Real loss
                disc_real_outputs = discriminator(aop_true, dolp_true, s0_true)
                real_labels = torch.ones_like(disc_real_outputs[0]).to(device)
                loss_D_real = (weight_aop * (adversarial_criterion(disc_real_outputs[0], real_labels)) +
                               weight_dolp * (adversarial_criterion(disc_real_outputs[1], real_labels)) +
                               weight_s0 * (adversarial_criterion(disc_real_outputs[2], real_labels)))/3

                # Fake loss
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                disc_fake_outputs = discriminator(aop_pred.detach(), dolp_pred.detach(), s0_pred.detach())
                fake_labels = torch.zeros_like(disc_fake_outputs[0]).to(device)
                loss_D_fake = (weight_aop * (adversarial_criterion(disc_fake_outputs[0], fake_labels)) +
                               weight_dolp * (adversarial_criterion(disc_fake_outputs[1], fake_labels)) +
                               weight_s0 * (adversarial_criterion(disc_fake_outputs[2], fake_labels)))/3
                
                loss_D = (loss_D_real + loss_D_fake) / 2

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()
            running_loss_D += loss_D.item()
            train_loss_D += loss_D.item()
            
            # Update generator
            optimizer_G.zero_grad()
            with autocast():
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                # Content loss
                content_loss = custom_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)                
                # Adversarial loss
                disc_outputs = discriminator(aop_pred, dolp_pred, s0_pred)
                adversarial_loss = (weight_aop * (adversarial_criterion(disc_outputs[0], real_labels)) +
                                    weight_dolp * (adversarial_criterion(disc_outputs[1], real_labels)) +
                                    weight_s0 * (adversarial_criterion(disc_outputs[2], real_labels)))/3
                #Perceptual loss
                perceptual_loss = (perceptual_criterion(aop_pred,aop_true) +
                                   perceptual_criterion(dolp_pred,dolp_true) + 
                                   perceptual_criterion(s0_pred,s0_true))/3
                w_adversarial = 1e-2
                w_perceptual = 1e-5
                loss_G = content_loss + w_adversarial * adversarial_loss + w_perceptual * perceptual_loss
              
            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()
            running_loss_G += loss_G.item()
            train_loss_G += loss_G.item()
            
            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Generator loss: {running_loss_G / 100:.4f}, Discriminator loss: {running_loss_D / 100:.4f}')
                running_loss_G = 0.0
                running_loss_D = 0.0
                
        # Calculate elapsed time and training loss
        elapsed_time = time.time() - start_time
        writer.add_scalar('Training Time', elapsed_time, epoch + 1)
        
        avg_loss_G = train_loss_G / len(train_loader)
        avg_loss_D = train_loss_D / len(train_loader)
        writer.add_scalars('Loss', {'Generator loss': avg_loss_G, 'Discriminator loss': avg_loss_D}, global_step=epoch + 1)
        torch.cuda.empty_cache()
        
        # Validation
        generator.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_psnr_aop = 0.0
        val_psnr_dolp = 0.0
        val_psnr_s0 = 0.0
        psnr_val_s0 = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
        psnr_val_dolp = torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(device)
        psnr_val_aop = torchmetrics.PeakSignalNoiseRatio(data_range=torch.pi/2).to(device)
        with torch.no_grad():
            for data, aop, dolp, s0 in val_loader:
                inputs = data.to(device)
                aop_true = aop.to(device)
                dolp_true = dolp.to(device)
                s0_true = s0.to(device)
                aop_pred, dolp_pred, s0_pred = generator(inputs)
                
                loss = (custom_criterion(s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true)
                        + 0.1 * w_perceptual * (perceptual_criterion(aop_pred,aop_true) +
                                   perceptual_criterion(dolp_pred,dolp_true) + 
                                   perceptual_criterion(s0_pred,s0_true)/3))
                val_loss += loss.item()
                
                psnr_aop = psnr_val_aop(aop_pred, aop_true)
                psnr_dolp = psnr_val_dolp(dolp_pred, dolp_true)
                psnr_s0 = psnr_val_s0(s0_pred, s0_true)
                
                val_psnr_aop += psnr_aop.item()
                val_psnr_dolp += psnr_dolp.item()
                val_psnr_s0 += psnr_s0.item()
                
                psnr_avg = (psnr_aop + psnr_dolp + psnr_s0) / 3
                val_psnr += psnr_avg.item()
                
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        
        val_psnr_aop /= len(val_loader)
        val_psnr_dolp /= len(val_loader)
        val_psnr_s0 /= len(val_loader)
        
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}')
        writer.add_scalar('Validation Loss', val_loss, epoch + 1)
        writer.add_scalar('Validation PSNR', val_psnr, epoch + 1)
        writer.add_scalars('PSNR', {'PSNR of AoP': val_psnr_aop, 'PSNR of DoLP': val_psnr_dolp, 
                                    'PSNR of S0': val_psnr_s0}, global_step=epoch + 1)
        torch.cuda.empty_cache()

        # Save checkpoint
        torch.save({
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict()
        }, checkpoint_path)
        print('Checkpoint saved.')

        # Save best model based on validation
        if savebest:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'epoch': best_epoch,
                    'val_loss': val_loss,
                }, best_checkpoint_path)
                print('Best checkpoint saved.')
                
        # Save best model based on PSNR
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch + 1
                best_checkpoint_path = checkpoint_path.replace('.pth', '_best_psnr.pth')
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'epoch': best_epoch,
                    'psnr': best_psnr,
                }, best_checkpoint_path)
                print(f'Best checkpoint based on PSNR saved at epoch {best_epoch}, PSNR: {best_psnr:.4f}')
                
        torch.cuda.empty_cache()
    writer.close()
    print('Finished Training')
    
if __name__ == "__main__":
    lr = 0.001
    num_epochs = 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = Generator()
    discriminator = Discriminator()

    batch_size = 50
    weight_decay = 1e-4

    train_file_path = 'data/your_traing_dataset.h5'
    test_file_path = 'data/your_test_dataset.h5'
    train_dataset = MyDataset(train_file_path)
    val_dataset = MyDataset(test_file_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10, num_workers=8, pin_memory=True, shuffle=False)
    
    train(generator=generator, discriminator=discriminator, train_loader=train_loader, val_loader=val_loader, num_epochs=num_epochs, learning_rate=lr, weight_decay=weight_decay, device=device)
    