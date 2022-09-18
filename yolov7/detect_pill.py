import argparse
import time
from pathlib import Path
import numpy as np
import os
import cv2
import csv
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import warnings
from PIL import Image
import cv2
import os
import json
import Levenshtein
import re
warnings.simplefilter("ignore")
import tqdm

def load_ids(root_path_label):
    ids_filter = dict()

    files = os.listdir(root_path_label)

    for file in files:
        id_file = re.findall('[0-9]+',file)[0]
        
        path_file = os.path.join(root_path_label, file)

        ann = open(path_file,'r').read()

        classes = json.loads(ann)
        
        ids_filter[id_file] = classes
    return ids_filter

def detect(args,save_img=False):
    



    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = Path(increment_path(Path(opt.project), exist_ok=True)) 
    if os.path.exists(save_dir):
        os.rmdir(save_dir)
    (save_dir / 'crop_img').mkdir(parents=True, exist_ok=True)  # make dir

    csv_path = os.path.join(args.output_csv,'output.csv')
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with open(csv_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        data = ['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']
        writer.writerow(data)
    save_dir_crop = os.path.join(save_dir, 'crop_img')
    os.makedirs(save_dir_crop,exist_ok = True)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16


    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    id = 0
    for path, img, im0s, vid_cap in tqdm.tqdm(dataset):
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
         
  
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if False:   #scale box 
                        scale=0.025
                        w = xyxy[2] - xyxy[0]
                        h = xyxy[3] - xyxy[1]
                        scale_x = w*scale
                        scale_y = h*scale
                        xyxy[0] += scale_x
                        xyxy[2] -= scale_x
                        xyxy[1] += scale_y
                        xyxy[3] -= scale_y

                    x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            
                    img_crop = im0[int(y1):int(y2),int(x1):int(x2)]
                    new_img_name=f'{id}_{os.path.basename(path)}'
                    path_crop = os.path.join(save_dir_crop,new_img_name)
                    cv2.imwrite(path_crop,img_crop)
                    id+=1
         
                    class_id = 107
                    conf_cls=0
                    
                    with open(csv_path, 'a', encoding='UTF8') as f:
                        writer = csv.writer(f)
                        data = [new_img_name, class_id, conf.item(),  x1,y1,x2,y2] #cls.item()

                        writer.writerow(data)
                    # if save_img or view_img:  # Add bbox to image
                    #     label = f'{conf:.2f}'
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    #     cv2.putText(im0, f'{filter_id}', (100,100), cv2.FONT_HERSHEY_PLAIN, 5, (0,255,0), 2)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
                
                    # print(f" The image with the result is saved in: {save_path}")
    
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_csv',  type=str, default='../output/detect', help='')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')


    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect(opt)
