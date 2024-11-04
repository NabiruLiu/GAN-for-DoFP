import numpy as np
from .utils import dolp, aop
import h5py
import cv2
import torch
import torchvision.transforms.functional as F
import os

def load_images(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.bmp'):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)

            if image is not None:
                try:
                    angle = int(filename.split('_')[1].split('.')[0])
                except ValueError:
                    try:
                        angle = int(filename.split('_')[2].split('.')[0])
                    except ValueError:
                        print(f"Invalid angle in filename: {filename}")
                        continue
                images[angle] = torch.from_numpy(image).float() / 255.0
    return images

def create_labels(root_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dofp_list = []
    s0_list = []
    dolp_list = []
    aop_list = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == os.path.basename(root_folder):
                continue
            dir_fullpath = os.path.join(root_folder, dirname)
            images = load_images(dir_fullpath)

            if len(images) == 4 and 0 in images and 45 in images and 90 in images and 135 in images:
                dofp = torch.zeros_like(images[0].to(device))
                img_aop = torch.zeros_like(images[0].to(device))
                img_dolp = torch.zeros_like(images[0].to(device))
                img_s0 = torch.zeros_like(images[0].to(device))
                dofp[0::2, 0::2] = images[0][0::2, 0::2]
                dofp[0::2, 1::2] = images[45][0::2, 1::2]
                dofp[1::2, 1::2] = images[90][1::2, 1::2]
                dofp[1::2, 0::2] = images[135][1::2, 0::2]
                img_aop = aop(images[0], images[45], images[90], images[135])
                img_dolp = dolp(images[0], images[45], images[90], images[135])
                img_dolp[img_dolp > 1] = 1
                img_s0 = 1/2 * (images[0] + images[45] + images[90] + images[135])
                # normalize to range (0,1)
                img_s0 = 0.5 * img_s0
                # print(torch.max(img_dolp))
                # print(dofp.shape)
                dofp_list.append(dofp)
                aop_list.append (img_aop)
                dolp_list.append(img_dolp)
                s0_list.append(img_s0)

    dofp_tensor = torch.stack(dofp_list,dim=0).unsqueeze(1)
    s0_tensor = torch.stack(s0_list, dim=0).unsqueeze(1)
    dolp_tensor = torch.stack(dolp_list, dim=0).unsqueeze(1)
    aop_tensor = torch.stack(aop_list, dim=0).unsqueeze(1)
        
    return dofp_tensor, s0_tensor, aop_tensor, dolp_tensor

# Generate patches
def slice_and_save_to_h5(dofp_tensor, s0_tensor, aop_tensor, dolp_tensor, label_path, patch_size, stride, slice=True):
    data_patches = []
    s0_patches = []
    aop_patches = []
    dolp_patches = []
    data_list = []
    labels_list = []
    if slice:
        for i in range(dofp_tensor.shape[0]):
            dofp = dofp_tensor[i]
            s0 = s0_tensor[i]
            aop = aop_tensor[i]
            dolp = dolp_tensor[i]
            # print(dofp.shape)
            for y in range(0, dofp.shape[1] - patch_size + 1, stride):
                for x in range(0, dofp.shape[2] - patch_size + 1, stride):
                    data_patches.append(dofp[:, y:y+patch_size, x:x+patch_size])
                    s0_patches.append(s0[:, y:y+patch_size, x:x+patch_size])
                    aop_patches.append(aop[:, y:y+patch_size, x:x+patch_size])
                    dolp_patches.append(dolp[:, y:y+patch_size, x:x+patch_size])
            
            # Process edge region
            if y + patch_size < dofp.shape[1]:
                data_patches.append(dofp[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                s0_patches.append(s0[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                aop_patches.append(aop[:, dofp.shape[1] - patch_size:, x:x+patch_size])
                dolp_patches.append(dolp[:, dofp.shape[1] - patch_size:, x:x+patch_size])

            if x + patch_size < dofp.shape[2]:
                data_patches.append(dofp[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                s0_patches.append(s0[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                aop_patches.append(aop[:, y:y+patch_size, dofp.shape[2] - patch_size:])
                dolp_patches.append(dolp[:, y:y+patch_size, dofp.shape[2] - patch_size:])
        
        data_patches = torch.stack(data_patches)
        s0_patches = torch.stack(s0_patches)
        aop_patches = torch.stack(aop_patches)
        dolp_patches = torch.stack(dolp_patches)
        
        data_list.append(data_patches)
        labels_list.append((aop_patches, dolp_patches, s0_patches))
    
    else:
        data_patches = dofp_tensor
        s0_patches = s0_tensor
        aop_patches = aop_tensor
        dolp_patches = dolp_tensor
        
        data_list.append(data_patches)
        labels_list.append((aop_patches, dolp_patches, s0_patches))
            
    with h5py.File(label_path, 'w') as hf:
        data_group = hf.create_group('data')
        labels_group = hf.create_group('labels')

        for i, data in enumerate(data_patches):
            data_group.create_dataset(f'data_{i}', data=data.squeeze(0).cpu().numpy())
        
        for i, (aop, dolp, s0) in enumerate(zip(aop_patches, dolp_patches, s0_patches)):
            label_group = labels_group.create_group(f'label_{i}')
            label_group.create_dataset('aop', data=aop.squeeze(0).cpu().numpy())
            label_group.create_dataset('dolp', data=dolp.squeeze(0).cpu().numpy())
            label_group.create_dataset('s0', data=s0.squeeze(0).cpu().numpy())
            
    print('Finished!')
            
if __name__ == '__main__': 
    root_path = 'Your data path'
    label_path = 'Your save path.h5'
    dofp_tensor, s0_tensor, aop_tensor, dolp_tensor = create_labels(root_path)
    patch_size = 100
    coincide = 20
    stride = patch_size - coincide
    slice_and_save_to_h5(dofp_tensor, s0_tensor, aop_tensor, dolp_tensor, label_path, patch_size, stride, slice=True)
