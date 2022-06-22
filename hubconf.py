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
        arch="uniformer_small",
        url="https://drive.google.com/u/0/uc?id=1-uepH3Q3BhTmWU6HK-sGAGQC_MpfIiPD",
        filename=os.path.join(_default_cache_dir, "uniformer_small_in1k.pth"),
        md5="f122ab0dde94b1d73ac00d2c5359a610",
        input_size=224
    ),
    "uniformer_small_plus_in1k": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=10IN5ULcjz0Ld_lDokkTGOSmRFXLzUkEs",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_in1k.pth"),
        md5="9a9e04132baa7c61b50611b636ed4bea",
        input_size=224
    ), 
}   

def _transform(input_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    size = int((256 / 224) * input_size)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform
 
def load_state_dict_from_gdrive(url, filename, hashid):    
    weights_file = gdown.cached_download(url, filename, md5=hashid)
    checkpoint = torch.load(weights_file)

    if 'model' in checkpoint:
      state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
      state_dict = checkpoint['state_dict']
    else:
      state_dict = checkpoint
    
    return state_dict, weights_file

def load_weights(model, url, hashid, filename, verbose=True):

    state_dict, weights_file = load_state_dict_from_gdrive(url, filename, hashid)
    
    if verbose: 
        print(f"==> loading checkpoint: {Path(weights_file).name}")
    
    msg = model.load_state_dict(state_dict, strict=True)
    model.hashid = hashid
    model.weights_file = weights_file
    
    if verbose: 
        print(f"==> state loaded: {msg}")
    
    return model

# -------------------------------------
#  Classification Models
# -------------------------------------

def uniformer_small_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    arch = _cfg[model_name]['arch']
    model = cls_models.__dict__[arch](pretrained=False, **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=_cfg[model_name]['input_size'])
    
    return model, transform
