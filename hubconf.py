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
        input_size=224,
        task="in1k"
    ),
    "uniformer_small_plus_in1k": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=10IN5ULcjz0Ld_lDokkTGOSmRFXLzUkEs",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_in1k.pth"),
        md5="9a9e04132baa7c61b50611b636ed4bea",
        input_size=224,
        task="in1k"
    ),    
    "uniformer_base_in1k": dict(
        arch="uniformer_base",
        url="https://drive.google.com/u/0/uc?id=1-wT39QazTGELxgrQIu6J12D3qcla3hui",
        filename=os.path.join(_default_cache_dir, "uniformer_base_in1k.pth"),
        md5="5be77ad1f3506a9c3edcabcb71a1de6b",
        input_size=224,
        task="in1k"
    ),  
    "uniformer_base_ls_in1k": dict(
        arch="uniformer_base_ls",
        url="https://drive.google.com/u/0/uc?id=10FnwzMVgGL8bPO3EMZ7oG4vDvqnPoBK5",
        filename=os.path.join(_default_cache_dir, "uniformer_base_ls_in1k.pth"),
        md5="3491dddcda1babb9ea798bbbf6942a40",
        input_size=224,
        task="in1k"
    ),      
    "uniformer_small_plus_dim64_in1k": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=178ipRvMBKeAP_Fzy6fEr2HQkIQWdgEY6",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_dim64_in1k.pth"),
        md5="3a5aefe567d6e97149d60af91aa958da",
        input_size=224,
        task="in1k"
    ),   
    "uniformer_small_tl_224": dict(
        arch="uniformer_small",
        url="https://drive.google.com/u/0/uc?id=16wjfeyqQFZ2x9W91EXf0Py1bfv4Fydfr",        
        filename=os.path.join(_default_cache_dir, "uniformer_small_tl_224.pth"),
        md5="b2a33929f20a5bbde0d237cc03f1d4fc",
        input_size=224,
        task="in1k"
    ),   
    "uniformer_small_plus_tl_224": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=16rgtEVEeX3jVTQaiLsojBQP5sRaj_2aZ",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_tl_224.pth"),
        md5="83abb926261d1f0dbacaae239a53360c",
        input_size=224,
        task="in1k"
    ),    
    "uniformer_base_tl_224": dict(
        arch="uniformer_base",
        url="https://drive.google.com/u/0/uc?id=16nsTjawiYZuoCHBRarr58zA7vCFcmQmL",
        filename=os.path.join(_default_cache_dir, "uniformer_base_tl_224.pth"),
        md5="1ffe1f83a214b4e0173a0df6ba344fb3",
        input_size=224,
        task="in1k"
    ),
    # "uniformer_large_ls_tl_224": dict(
    #     arch="uniformer_large",
    #     url="https://drive.google.com/u/0/uc?id=173e2qaJhOwbuC_DbNQSH47Oo_04mSfDj",
    #     filename=os.path.join(_default_cache_dir, "uniformer_large_ls_tl_224.pth"),
    #     md5="a26ef63a4ab2b1eaf59a106724666273",
    #     input_size=224,
    #     task="in1k"
    # ),       
    "uniformer_small_dim64_tl_224": dict(
        arch="uniformer_small",
        url="https://drive.google.com/u/0/uc?id=17NC5qqAWCT2gI_jY_cJnFTOqRQTSwsoz",
        filename=os.path.join(_default_cache_dir, "uniformer_small_dim64_tl_224.pth"),
        md5="771bf024006256c8406892c6159d4384",
        input_size=224,
        task="in1k"
    ),  
    "uniformer_small_plus_dim64_tl_224": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=17OfoDWt2_nA0BYN0OuLPpG8mHJDTluQQ",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_dim64_tl_224.pth"),
        md5="723f198c629e2cd06630f380b3f68ec1",
        input_size=224
    ),      
    "uniformer_base_dim64_tl_224": dict(
        arch="uniformer_base",
        url="https://drive.google.com/u/0/uc?id=17MneG9CnZG6zBvXYUr4WUXoTYc4rjJj5",
        filename=os.path.join(_default_cache_dir, "uniformer_base_dim64_tl_224.pth"),
        md5="a1c1abf5bcac177c16114229132b4327",
        input_size=224,
        task="in1k"
    ),   
    "uniformer_small_tl_384": dict(
        arch="uniformer_small",
        url="https://drive.google.com/u/0/uc?id=16p4CXzuXC5J4_SJK67dsvnc8gTKmtKN1",
        filename=os.path.join(_default_cache_dir, "uniformer_small_tl_384.pth"),
        md5="c9cae5878b5837215a976e91021db95d",
        input_size=384,
        task="in1k"
    ),
    "uniformer_small_plus_tl_384": dict(
        arch="uniformer_small_plus",
        url="https://drive.google.com/u/0/uc?id=16wJT87vTc43Dt1q2sXrxZRUuEdrNdFdk",
        filename=os.path.join(_default_cache_dir, "uniformer_small_plus_tl_384.pth"),
        md5="e017ce278be2a05074c8d8b7c9d1a31a",
        input_size=384,
        task="in1k"
    ),    
    "uniformer_base_tl_384": dict(
        arch="uniformer_base",
        url="https://drive.google.com/u/0/uc?id=16kZlIarIwf9ldkCdHVKTQfFmDgfkucQ7",
        filename=os.path.join(_default_cache_dir, "uniformer_base_tl_384.pth"),
        md5="2dcee4dd928a22c296f79f1d41f6ffba",
        input_size=384,
        task="in1k"
    ),  
    # "uniformer_large_ls_tl_384": dict(
    #     arch="uniformer_large_ls",
    #     url="https://drive.google.com/u/0/uc?id=174rcA6rNzYVG9Ya9ik-NwTGoxW1M79ez",
    #     filename=os.path.join(_default_cache_dir, "uniformer_large_ls_tl_384.pth"),
    #     md5="03a3bfc20f40d272d13a3e02909074e4",
    #     input_size=384,
    #     task="in1k"
    # )             
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
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    assert len(missing_keys)==0, f"Oops, missing keys: {missing_keys}"
    if len(unexpected_keys) > 0:
        assert unexpected_keys == ['aux_head.weight', 'aux_head.bias'], f"Oops, unexpected unexpected_keys: {unexpected_keys}"
        msg = f"<unexpected_keys: {unexpected_keys}>"
    else:
        msg = "<All keys matched successfully>"
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
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_plus_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_base_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_base_ls_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_plus_dim64_in1k(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_plus_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_base_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_dim64_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_plus_dim64_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_base_dim64_tl_224(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_tl_384(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_small_plus_tl_384(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform

def uniformer_base_tl_384(pretrained=True, verbose=True, **kwargs):
    model_name = inspect.stack()[0][3]
    cfg = _cfg[model_name]
    model = cls_models.__dict__[cfg['arch']](pretrained=False, img_size=cfg['input_size'], **kwargs)
    if pretrained:
        model = load_weights(model, cfg['url'], cfg['md5'], cfg['filename'], verbose=verbose)

    transform = _transform(input_size=cfg['input_size'])
    
    return model, transform












