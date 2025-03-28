from PIL import Image
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F

def load_image(image_path):
    image_np = np.array(Image.open(image_path))
    if image_np.ndim == 2:
        image_np = np.tile(image_np[:, :, None], 3)
    return image_np

def resize_image(image_np, size=(256, 256), resample=3):
    return np.array(Image.fromarray(image_np).resize((size[1], size[0]), resample=resample))

def preprocess_image(image_rgb, target_size=(256, 256)):
    resized_image = resize_image(image_rgb, size=target_size)

    lab_original = color.rgb2lab(image_rgb)
    lab_resized = color.rgb2lab(resized_image)

    l_original = lab_original[:, :, 0]
    l_resized = lab_resized[:, :, 0]

    original_tensor = torch.Tensor(l_original).unsqueeze(0).unsqueeze(0)
    resized_tensor = torch.Tensor(l_resized).unsqueeze(0).unsqueeze(0)

    return original_tensor, resized_tensor

def postprocess_tensor(original_l_tensor, ab_tensor, mode='bilinear'):
    original_hw = original_l_tensor.shape[2:]
    output_hw = ab_tensor.shape[2:]

    if original_hw != output_hw:
        ab_tensor_resized = F.interpolate(ab_tensor, size=original_hw, mode=mode)
    else:
        ab_tensor_resized = ab_tensor

    lab_result = torch.cat((original_l_tensor, ab_tensor_resized), dim=1)
    rgb_result = color.lab2rgb(lab_result[0].detach().cpu().numpy().transpose((1, 2, 0)))

    return np.clip(rgb_result, 0, 1)

