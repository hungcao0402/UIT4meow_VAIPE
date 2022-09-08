import json 
import glob
import cv2
import os
import shutil
from tqdm import tqdm

root = '/data/public_train/prescription'
label_paths = root + '/label/*'
image_paths = root + '/image/'
class_list = {'date':0,'drugname':1,'diagnose':2,'usage':3,'quantity':4,'other':5}

def xyxy2xywh(box, img_shape):
    x,y,x1,y1 = box
    x = (x+x1)/2
    y = (y+y1)/2
    w = (x1 - x)*2
    h = (y1 - y)*2
    return x/img_shape[1], y/img_shape[0], w/img_shape[1], h/img_shape[0]

def main():
    os.makedirs('./dataset/pres/yolov7/images/train', exist_ok = True)
    os.makedirs('./dataset/pres/yolov7/images/val', exist_ok = True)
    os.makedirs('./dataset/pres/yolov7/labels/train', exist_ok = True)
    os.makedirs('./dataset/pres/yolov7/labels/val', exist_ok = True)

    train_val_ratio = 0.9
    train_len = 0.9*len(glob.glob(label_paths))
    count = 0
    for file in tqdm(glob.glob(label_paths)):
        f = open(file, 'r')
        ann = json.load(f)
        img_path = file.replace('json', 'png').replace('label','image')
        if count < train_len:
            shutil.copy2(img_path,f'dataset/pres/yolov7/images/train/{os.path.basename(img_path)}')
        else:
            shutil.copy2(img_path,f'dataset/pres/yolov7/images/val/{os.path.basename(img_path)}')


        img = cv2.imread(img_path)
        for i in ann:
            if i['label'] == 'drugname':
                box = i['box']

                x,y,w,h = xyxy2xywh(box,img.shape)
                if count < train_len:
                    with open(f'dataset/pres/yolov7/labels/train/{os.path.basename(file).replace("json","txt")}', 'a') as f:
                            f.write(f'0 {x} {y} {w} {h}\n')
                else:
                    with open(f'dataset/pres/yolov7/labels/val/{os.path.basename(file).replace("json","txt")}', 'a') as f:
                            f.write(f'0 {x} {y} {w} {h}\n')
        count+=1
main()

