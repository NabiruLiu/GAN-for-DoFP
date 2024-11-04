import torch
import numpy as np
from model import  Generator
import cv2
import os

def process_dofp(checkpoint_path, input_image_path, output_dir):
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator()  # Ensure your Generator class is defined/imported
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['generator_state_dict'])
    model = model.to(device)
    model.eval()

    # Load and normalize the image data to [0, 1]
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    img = img / 255.0  # Normalize 8bit image to [0, 1], you should adjust this to adapt your data.
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        aop, dolp, s0 = model(img)

    # Move outputs to CPU
    aop = aop.cpu()
    aop = np.mod(aop * 180 - 90, 180) / 180
    dolp = dolp.cpu()
    s0 = s0.cpu()

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert and save outputs as 8-bit PNG, you can adjust data type for your need.
    aop_np = aop.squeeze().numpy()
    aop_np = np.clip(aop_np * 255, 0, 255).astype(np.uint8)  # Scale and convert to 8-bit
    cv2.imwrite(os.path.join(output_dir, 'aop.png'), aop_np)

    dolp_np = dolp.squeeze().numpy()
    dolp_np = np.clip(dolp_np * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 'dolp.png'), dolp_np)
    
    s0_np = s0.squeeze().numpy()
    s0_np = np.clip(s0_np * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, 's0.png'), s0_np)
    print('Finished.')

if __name__ == '__main__':
    checkpoint_path = 'ckpt/your_checkpoint.pth'
    input_image_path = 'input/your_image.png'
    output_dir = 'output/'
    process_dofp(checkpoint_path, input_image_path, output_dir)
