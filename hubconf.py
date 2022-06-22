'''
    
'''

import os
import torch
import torchvision

import timm
import image_classification.models as cls_models

dependencies = ["timm", "torch", "torchvision"]

def _transform(resize=256, crop_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize),
        torchvision.transforms.CenterCrop(crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)    
    ])
    
    return transform
    
def uniformer_small_in1k(pretrained=True, **kwargs):
    model = timm.load_model("uniformer_small", pretrained=False, **kwargs)
    if pretrained:
      pass
#         state_dict = torch.hub.load_state_dict_from_url(
#             url="https://dl.fbaipublicfiles.com/vicreg/resnet50.pth",
#             map_location="cpu",
#             file_name="resnet50-c843e76524.pth",
#             check_hash=True
#         )
#         model.load_state_dict(state_dict, strict=True)
#         model.hashid = 'c843e76524'
#         model.weights_file = os.path.join(torch.hub.get_dir(), "checkpoints", "resnet50-c843e76524.pth")
        
    transform = _transform()
    
    return model, transform
