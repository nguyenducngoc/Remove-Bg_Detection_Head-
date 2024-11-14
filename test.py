import cv2
import gradio as gr
import os
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import gdown
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from isnet import ISNetDIS
import sys
import time
import tensorflow as tf

from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

from data_loader_cache import normalize, im_reader, im_preprocess 

#Helpers
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
class GOSNormalize(object):

    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image


transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])

def load_image(im_path, hypar):
    im = im_reader(im_path)
    im, im_shp = im_preprocess(im, hypar["cache_size"])
    im = torch.divide(im,255.0)
    shape = torch.from_numpy(np.array(im_shp))
    return transform(im).unsqueeze(0), shape.unsqueeze(0) 


def build_model(hypar,device):
    net = hypar["model"]#GOSNETINC(3,1)

    # convert to half precision
    if(hypar["model_digit"]=="half"):
        net.half()
        for layer in net.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()

    net.to(device)

    if(hypar["restore_model"]!=""):
        net.load_state_dict(torch.load(hypar["model_path"]+"/"+hypar["restore_model"], map_location=device))
        net.to(device)
    net.eval()  
    return net

    
def predict(net,  inputs_val, shapes_val, hypar, device):
    '''
    Given an Image, predict the mask
    '''
    net.eval()

    if(hypar["model_digit"]=="full"):
        inputs_val = inputs_val.type(torch.FloatTensor)
    else:
        inputs_val = inputs_val.type(torch.HalfTensor)

  
    inputs_val_v = Variable(inputs_val, requires_grad=False).to(device) # wrap inputs in Variable
   
    ds_val = net(inputs_val_v)[0] # list of 6 results

    pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

    ## recover the prediction spatial size to the orignal image size
    pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),(shapes_val[0][0],shapes_val[0][1]),mode='bilinear'))

    ma = torch.max(pred_val)
    mi = torch.min(pred_val)
    pred_val = (pred_val-mi)/(ma-mi) # max = 1

    if device == 'cuda': torch.cuda.empty_cache()
    return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need
    
# Set Parameters
hypar = {} # paramters for inferencing


hypar["model_path"] ="./saved_models" ## load trained weights from this path
hypar["restore_model"] = "isnet.pth" ## name of the to-be-loaded weights
hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision

##  choose floating point accuracy --
hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
hypar["seed"] = 0

hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size

## data augmentation parameters ---
hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

hypar["model"] = ISNetDIS()

 # Build Model
net = build_model(hypar, device)

#detection
PATH_TO_CKPT_HEAD = 'HEAD_DETECTION_300x300_ssd_mobilenetv2.pb'
head_detector = FROZEN_GRAPH_HEAD(PATH_TO_CKPT_HEAD)

def inference(image):
    image_path = image
    
    image_tensor, orig_size = load_image(image_path, hypar) 
    mask = predict(net, image_tensor, orig_size, hypar, device)
    
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image).convert("RGB")
    
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    
    im_array = np.array(im_rgba)
    im_height, im_width = im_array.shape[:2]
    
    _, heads = head_detector.run(im_array, im_width, im_height)
    
    if heads:
        total_height = sum(head['height'] for head in heads)
        max_width = max(head['width'] for head in heads)
        combined_crop = Image.new('RGBA', (max_width, total_height))
        
        y_offset = 0
        for head in heads:
            left = head['left']
            top = head['top']
            right = head['right']
            bottom = head['bottom']
            
            # Crop from im_rgba
            head_crop = im_rgba.crop((left, top, right, bottom))
            
            # Paste onto combined image
            combined_crop.paste(head_crop, (0, y_offset))
            y_offset += head['height']
            
        im_crop_pil = combined_crop
    else:
 
        im_crop_pil = Image.new('RGBA', (100, 100), (0, 0, 0, 0))

    return [im_rgba, pil_mask, im_crop_pil]


interface = gr.Interface(
    fn=inference,
    inputs=gr.Image(type='filepath'),
    outputs=[
        gr.Image(type='filepath', format="png"),  # im_rgba - masked image
        gr.Image(type='filepath', format="png"),  # pil_mask - alpha mask
        gr.Image(type='filepath', format="png")   # im_crop_pil - cropped heads from im_rgba
    ],
    flagging_mode="never",
    cache_mode="lazy",
).queue().launch(show_error=True)