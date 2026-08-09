import torch
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.nn import functional as F
from transformers import ViTModel

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def pil_to_n1_1_tensor(img):
    '''
    Convert a PIL image to a tensor in [-1, 1] range
    '''
    tensor = torch.from_numpy(np.array(img)) / 255 # [0,1]
    tensor = tensor * 2 - 1 # [-1,1]

    assert tensor.max() <= 1 and tensor.min() >= -1, ( 
                f'Tensor range must be in [-1, 1], is [{tensor.min()}, {tensor.max()}]')
    return tensor

# LPIPS needs the images to be in the [-1, 1] range.
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', reduction='mean')
def get_lpips(im1: Image.Image, 
              im2: Image.Image
              ):
    t_im1 = pil_to_n1_1_tensor(im1)
    t_im2 = pil_to_n1_1_tensor(im2)
    return lpips(t_im1, t_im2)


# DINO Transforms
T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# Load DINO ViT-S/16
dino = ViTModel.from_pretrained('facebook/dino-vits16')


def get_dinoscore(imgs: list[Image.Image]):
    # https://github.com/google/dreambooth/issues/3 (from dreambooth repo)
    images = [T(im) for im in imgs]

    inputs = torch.stack(images) # (2, 3, 224, 224). Batchsize = 2
    # Get DINO features
    with torch.no_grad():
        outputs = dino(inputs)

    last_hidden_states = outputs.last_hidden_state # ViT backbone features
    emb_img1, emb_img2 = last_hidden_states[0, 0], last_hidden_states[1, 0] # Get cls token (0-th token) for each img
    metric = F.cosine_similarity(emb_img1, emb_img2, dim=0).item()
    return metric
