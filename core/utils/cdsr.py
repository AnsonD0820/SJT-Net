import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np

input_folder = "/home/input_images"
output_folder = "/home/output_reconstructed_images"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # list
        filepath = os.path.join(input_folder, filename)

        # load_images
        image = cv2.imread(filepath)

        # To_Tensor
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0

        # Gaussian_downsample
        # Gaussiansampled_image = cv2.pyrDown(image)

        # Gaussian_upsample
        Gaussiansampled_image = cv2.pyrUp(image)

        Gaussiansampled_tensor = torch.from_numpy(Gaussiansampled_image.transpose((2, 0, 1))).float() / 255.0
        print(Gaussiansampled_tensor.shape)

        # Bilinear_downsample
        # interpolate_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False)

        # Bilinear_upsample
        interpolate_tensor = F.interpolate(image_tensor.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False)

        # fusion
        final_image = torch.mean(torch.stack([Gaussiansampled_tensor, interpolate_tensor.squeeze(0)]), dim=0).numpy()
        final_image = final_image.transpose((1, 2, 0)) * 255.0

        # save_list
        output_filepath = os.path.join(output_folder, filename)

        # save_images
        cv2.imwrite(output_filepath, final_image)

print("Done")