import torch
from timm.models import  mobilevitv2_075, beit_base_patch16_224,mobilevit_xxs,edgenext_xx_small, vit_tiny_patch16_224, vit_base_patch16_224
from timm.models.vision_transformer import VisionTransformer
#from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import sys

import onnx

def get_network(name):
    if "defo" in name  :
        from defo_model import DefoNet
        if 'dw' in name:
            use_dw = True
        else:
            use_dw = False
        _, sf, _ = name.strip().split('_') 
        sf = int(sf)
        model = DefoNet(3,2, scale_factor=sf, use_dw=use_dw)
        input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        model.eval()
    elif name in ['vit_base_16_112', 'vit_base_8_112']:
        if '16' in name:
            model = VisionTransformer(img_size=112,patch_size=16,num_classes=2)
        else:
            model = VisionTransformer(img_size=112,patch_size=8, num_classes=2)
        input_shape = [1, 3, 112, 112]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name in ['vit_tiny_224', 'vit_base_224']:
        if 'tiny' in name:
            model =vit_tiny_patch16_224(pretrained=False, num_classes=2)
        else:
            model =vit_base_patch16_224(pretrained=False, num_classes=2)
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name =='vit_tiny_16_112':
        model = VisionTransformer(img_size=112,patch_size=16,embed_dim=192, depth=12, num_heads=3,num_classes=2)
        input_shape = [1, 3, 112, 112]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name =='vit_tiny_8_112':
        model = VisionTransformer(img_size=112,patch_size=8,embed_dim=192, depth=12, num_heads=3,num_classes=2)
        input_shape = [1, 3, 112, 112]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name =='beit_base':
        model = beit_base_patch16_224(pretrained=False, num_classes=2)
        input_shape = [1, 3, 224, 224]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name =='mobilevit_108':
        model = mobilevitv2_075(pretrained=False, num_classes=2)
        input_shape = [1, 3, 108, 108]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    elif name =='mobilevit_256':
        model = mobilevitv2_075(pretrained=False, num_classes=2)
        input_shape = [1, 3, 256, 256]
        input_data = torch.randn(input_shape)
        input_name = "input0"
        shape_list = [(input_name, input_shape)]
    #print(model)
    return model, input_data

def cvt(model, dummy_input, output_file):
    input_names = [ "input0" ]
    output_names = [ "output1" ]
    torch.onnx.export(model, dummy_input, f"{output_file}.onnx", verbose=True, input_names=input_names, output_names=output_names,opset_version=11)

name = sys.argv[1]
model,di = get_network(name)
cvt(model, di, name)
