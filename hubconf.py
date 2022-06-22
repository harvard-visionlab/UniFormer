'''
    
'''

import os
import inspect
import torch
import torchvision
from pathlib import Path 

import hashlib
import gdown

import image_classification.models as cls_models

dependencies = ["torch", "torchvision", "gdown"]

_default_cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints") + os.path.sep

_cfg = {
    "uniformer_small_in1k": dict(
        url="https://drive.google.com/u/0/uc?id=1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD",
        filename=os.path.join(_default_cache_dir, "uniformer_small_in1k.pth"),
        md5="f122ab0dde94b1d73ac00d2c5359a610"
    )
}   

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform
    
def load_state_dict_from_gdrive(url, filename, hashid):    
    weights_file = gdown.cached_download(url, filename, md5=hashid)
    checkpoint = torch.load(weights_file)
    state_dict = checkpoint['model']
    return state_dict, weights_file

# -------------------------------------
#  Classification Models
# -------------------------------------

def uniformer_small_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    model = cls_models.uniformer_small(pretrained=False, **kwargs)
    if pretrained:
        url = _cfg[model_name]['url']
        hashid = _cfg[model_name]['md5']
        filename = _cfg[model_name]['filename']
        state_dict, weights_file = load_state_dict_from_gdrive(url, filename, hashid)
        if verbose: print(f"==> loading checkpoint: {Path(weights_file).name}")
        msg = model.load_state_dict(state_dict, strict=True)
        model.hashid = hashid
        model.weights_file = weights_file
        if verbose: print(f"==> state loaded: {msg}")

    transform = _transform()
    
    return model, transform
