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
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

from myFROZEN_GRAPH_HEAD import FROZEN_GRAPH_HEAD

from data_loader_cache import normalize, im_reader, im_preprocess 

#Helpers
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_face_outline(image, face_landmarks):
    FACE_OUTLINE_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67,
        109, 10
    ]
    
    points = []
    height, width = image.shape[:2]
    
    # Lấy tọa độ các điểm cho đường viền
    for idx in FACE_OUTLINE_INDICES:
        landmark = face_landmarks[idx]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append([x, y])
    
    points = np.array(points, dtype=np.int32)
    
    return points
    
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

def process_image(image_path):
    base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                         num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = detector.detect(mp_image)
    
    if detection_result.face_landmarks:
        face_outline = create_face_outline(image, detection_result.face_landmarks[0])
        result = np.zeros_like(image)
        cv2.fillPoly(result, [face_outline], (255, 255, 255))
        result = cv2.bitwise_and(image, result)
        result = Image.fromarray(result).convert('RGB')
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        return result

def save_transparent_image(image, save_path):
    """
    Lưu ảnh với nền trong suốt
    
    Parameters:
    - image: Ảnh đầu vào (numpy array)
    - save_path: Đường dẫn lưu file
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Kiểm tra xem ảnh có kênh alpha chưa
    if image.shape[-1] == 3:
        # Nếu ảnh có 3 kênh (BGR), thêm kênh alpha
        b, g, r = cv2.split(image)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        image = cv2.merge((b, g, r, alpha))
    elif image.shape[-1] == 4:
        # Chuyển từ RGBA sang BGRA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    
    cv2.imwrite(save_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

def putal_image(image):
    image_path = image
    image_tensor, orig_size = load_image(image_path, hypar) 
    mask = predict(net, image_tensor, orig_size, hypar, device)
    pil_mask = Image.fromarray(mask).convert('L')
    im_rgb = Image.open(image).convert("RGB")
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(pil_mask)
    
    return im_rgba

def save_mark(save_path, image):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    im_crop_array = np.array(image)
    rgb = cv2.cvtColor(im_crop_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, rgb)

def inference(image):
    #save
    save_path_crop = 'Remove-Bg/crop_transparent.png'
    save_path_mark = 'Remove-Bg/mark_transparent.png'
    save_path_final = 'Remove-Bg/final_transparent.png'
    save_test = 'Remove-Bg/test.png'
    save_test_mark = 'Remove-Bg/test_mark.png'

    im_rgba = putal_image(image)
    
    im_array = np.array(im_rgba)
    im_height, im_width = im_array.shape[:2]
    
    _, heads = head_detector.run(im_array, im_width, im_height)
    
    if heads:
        total_height = sum(head['height'] for head in heads)
        max_width = max(head['width'] for head in heads)
        combined_crop = Image.new('RGBA', (max_width, total_height), (0, 0, 0, 0))
        
        y_offset = 0
        for head in heads:
            left = head['left']
            top = head['top']
            right = head['right']
            bottom = head['bottom']
            
            head_crop = im_rgba.crop((left, top, right, bottom))
            head_crop_array = np.array(head_crop)
            alpha_channel = head_crop_array[:, :, 3]

            if not np.all(alpha_channel == 0):
                combined_crop.paste(head_crop, (0, y_offset), head_crop)
                y_offset += head['height']
        im_crop_pil = combined_crop
    #save crop
    im_crop_array = np.array(im_crop_pil)
    save_transparent_image(im_crop_array, save_path_crop)
    
    image_face = process_image(save_path_crop)

    save_path1 = 'Remove-Bg/mark.png'
    save_mark(save_path1, image_face)

    im_face_pil = putal_image(save_path1)
    im_face_array = np.array(im_face_pil)
    save_transparent_image(im_face_array, save_test)
    #bg
    image_1 = cv2.imread(save_path_crop)

    if image_1.shape[-1] != 4:
        b, g, r = cv2.split(image_1)
        alpha = np.ones(b.shape, dtype=b.dtype) * 255
        image_1 = cv2.merge((b, g, r, alpha))
    
    cut_percentage = 30
    height = image_1.shape[0]
    cut_height = int(height *(cut_percentage/100))

    image_1[height - cut_height:, :, 3] = 0
    
    save_transparent_image(image_1, save_path_mark)
    
    # im_head = putal_image(save_path_mark)
    # image = Image.open(save_path_mark)
    # background = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    background = Image.open(save_path_mark).convert('RGBA')
    # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    overlay = Image.open(save_test)
    overlay_sub = overlay.convert('RGBA')
    data = np.array(overlay_sub)
    r, g, b, a = data.T
    data = np.array([b, g, r, a])
    data = data.transpose()
    overlay_sub = Image.fromarray(data)
    
    result = Image.alpha_composite(overlay_sub, background)
    result.save(save_test_mark)
    
    image = cv2.imread(save_test_mark)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

interface = gr.Interface(
    fn=inference,
    inputs=gr.Image(type='filepath'),
    outputs=[
        gr.Image(type='filepath', format="png"),  # im_crop_pil - cropped heads from im_rgba
        # gr.Image(type='filepath', format="png"),  # image_face
        # gr.Image(type='filepath', format="png")   # final overlaid image
    ],
    flagging_mode="never",
    cache_mode="lazy",
).queue().launch(show_error=True)
