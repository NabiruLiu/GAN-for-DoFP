import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import Normalize
from torch.utils.data import Dataset
import math
import h5py
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import random

# Residual block
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=1))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

'''
GAN
'''
# Generator
class Generator(nn.Module):
    def __init__(self, num_blocks=[1, 1, 2]):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.layer1 = nn.Sequential(*resnet_block(64, 64, num_blocks[0], first_block=True))
        self.layer2 = nn.Sequential(*resnet_block(64, 128, num_blocks[1]))
        
        self.layer3_1 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        self.layer3_2 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        self.layer3_3 = nn.Sequential(*resnet_block(128, 32, num_blocks[2]))
        
        # Output layer
        # Depthwise separable convolution
        self.conv1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1_3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv2_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.conv2_3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)

        aop = self.layer3_1(x)
        aop = self.conv1_1(aop)
        aop = self.conv2_1(aop)
        
        dolp = self.layer3_2(x)
        dolp = self.conv1_2(dolp)
        dolp = self.conv2_2(dolp)
        
        s0 = self.layer3_3(x)
        s0 = self.conv1_3(s0)
        s0 = self.conv2_3(s0)
        
        return aop, dolp, s0

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Shared part
        self.shared = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Branches for aop, dolp, s0
        self.branch_aop = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2),
                                        nn.Sigmoid())
        self.branch_dolp = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2),
                                         nn.Sigmoid())
        self.branch_s0 = nn.Sequential(nn.Conv2d(256, 96, kernel_size=3, stride=1, padding=1),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Conv2d(96, 1, kernel_size=5, stride=1, padding=2),
                                       nn.Sigmoid())

    def forward(self, aop, dolp, s0):
        out_aop = self.shared(aop)
        out_aop = self.branch_aop(out_aop)

        out_dolp = self.shared(dolp)
        out_dolp = self.branch_dolp(out_dolp)

        out_s0 = self.shared(s0)
        out_s0 = self.branch_s0(out_s0)
        
        return out_aop, out_dolp, out_s0

'''
Dataset
'''
class MyDataset(Dataset):
    def __init__(self, file_path, transform=None):
        super(MyDataset, self).__init__()
        self.file_path = file_path
        self.transform = transform
        with h5py.File(self.file_path, 'r') as h5file:
            self.data_len = len(h5file['data'])
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as h5file:
            data = torch.from_numpy(h5file[f'data/data_{idx}'][...]).unsqueeze(0)
            aop = torch.from_numpy(h5file[f'labels/label_{idx}/aop'][...]).unsqueeze(0)
            dolp = torch.from_numpy(h5file[f'labels/label_{idx}/dolp'][...]).unsqueeze(0)
            s0 = torch.from_numpy(h5file[f'labels/label_{idx}/s0'][...]).unsqueeze(0)

        if self.transform:
            data, aop, dolp, s0 = self.transform(data, aop, dolp, s0)
        
        return data, aop, dolp, s0

# Data augmentation transform
def custom_transform(data, aop, dolp, s0):
    # Random horizontal flip
    if random.random() > 0.5:
        data = TF.hflip(data)
        aop = TF.hflip(aop)
        dolp = TF.hflip(dolp)
        s0 = TF.hflip(s0)

    # Random vertical flip
    if random.random() > 0.5:
        data = TF.vflip(data)
        aop = TF.vflip(aop)
        dolp = TF.vflip(dolp)
        s0 = TF.vflip(s0)

    # Random 0, 90, 180, 270 degree rotation
    angle = random.choice([0, 90, 180, 270])
    data = TF.rotate(data, angle)
    aop = TF.rotate(aop, angle)
    dolp = TF.rotate(dolp, angle)
    s0 = TF.rotate(s0, angle)

    return data, aop, dolp, s0
    
'''
Loss Functions
'''
# Content loss and physics loss
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, s0_pred, s0_true, dolp_pred, dolp_true, aop_pred, aop_true):
        # L1 loss
        loss_s0 = self.l1_loss(s0_pred, s0_true)
        loss_dolp = self.l1_loss(dolp_pred, dolp_true)
        loss_aop = self.l1_loss(aop_pred, aop_true)
        
        # Physics informed loss        
        Q_pred = dolp_pred * s0_pred * torch.cos(2 * (aop_pred * torch.pi - torch.pi/2)) # Return AoP from (0,1) to (-pi/2,pi/2)
        U_pred = dolp_pred * s0_pred * torch.sin(2 * (aop_pred * torch.pi - torch.pi/2))
        Q_true = dolp_true * s0_true * torch.cos(2 * (aop_true * torch.pi - torch.pi/2))
        U_true = dolp_true * s0_true * torch.sin(2 * (aop_true * torch.pi - torch.pi/2))        
        loss_Q = torch.mean(abs(Q_pred - Q_true))
        loss_U = torch.mean(abs(U_pred - U_true))
        physics_loss = loss_Q + loss_U

        # Total loss
        total_loss  = (0.1 * loss_s0 + 0.6 * loss_dolp + 0.3 * loss_aop)\
                    - 0.02 * SSIM(aop_pred,aop_true, data_range= math.pi/2)\
                    + 1 * physics_loss
        
        return total_loss

# Perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self, feature_layers=None, use_gpu=True):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features
        if feature_layers is None:
            self.feature_layers = [5, 10, 19]
        else:
            self.feature_layers = feature_layers
        
        self.vgg = nn.Sequential(*[self.vgg[i] for i in range(max(self.feature_layers) + 1)])
        self.vgg.eval()

        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.criterion = nn.MSELoss()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.vgg.to(self.device)
    
    def forward(self, inputs, targets):
        total_loss = 0
        for input, target in zip(inputs, targets):
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
            input = self.normalize(input)
            target = self.normalize(target)
            
            input = input.to(self.device)
            target = target.to(self.device)
            
            input_features = self.extract_features(input)
            target_features = self.extract_features(target)
            
            loss = 0
            for inp_feat, tgt_feat in zip(input_features, target_features):
                loss += self.criterion(inp_feat, tgt_feat)
            
            total_loss += loss
        
        return total_loss
    
    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        return features
 