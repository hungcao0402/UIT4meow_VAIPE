import os, sys
import torch, json
import numpy as np
import argparse

from main import build_model_main
from util.slconfig import SLConfig
from dataset import LoadImages
import cv2
import numpy as np
import csv
from tqdm import tqdm
def xyxy_norm2xyxy(boxes, shape):
    bbox = []
    W, H = shape
    for box in boxes:
        xmin, ymin, xmax, ymax = box.tolist()
        
        bbox.append([xmin*W, ymin*H, xmax*W, ymax*H])

    return bbox

def main(opt):
    save_dir_crop = os.path.join(opt.output,'crop_img')
    os.makedirs(save_dir_crop, exist_ok = True)
    csv_path_save = os.path.join(opt.output, 'output.csv')
    if os.path.exists(csv_path_save):
        os.remove(csv_path_save)

    with open(csv_path_save, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        data = ['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']
        writer.writerow(data)

    #build model
    args = SLConfig.fromfile(opt.model_config_path) 
    args.device = 'cuda' 
    args.backbone_dir='pretrained'
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(opt.model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    #Load data
    dataset = LoadImages(opt.source)

    

    thershold=opt.thershold
    #Detect
    with torch.no_grad():
        id=0

        for path, img0, img, shape in tqdm(dataset):
            output = model.cuda()(img[None].cuda())
            output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            scores = output['scores']
            select_mask = scores > thershold

            boxes = output['boxes'][select_mask]
            scores = scores[select_mask]

            boxes = xyxy_norm2xyxy(boxes, shape)

            img0 = cv2.cvtColor(np.array(img0),cv2.COLOR_BGR2RGB)
                
            for box,score in zip(boxes,scores):
                x1, y1, x2, y2 = box
                img_crop = img0[int(y1):int(y2), int(x1):int(x2)]
                new_img_name=f'{id}_{os.path.basename(path)}'
                path_crop = os.path.join(save_dir_crop,new_img_name)
                id_test = os.path.basename(path).split('_')[2]
                
                try:
                    
                    cv2.imwrite(path_crop,img_crop)
                    #create coco format
                except:
                    continue
                id+=1
                    
                with open(csv_path_save, 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    data = [new_img_name, 0, score.item(),  x1,y1,x2,y2]
                    writer.writerow(data)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='root pill image')
    parser.add_argument('--output', type=str, default='../output/detect', help='path save detect result')
    parser.add_argument('--thershold', type=float, default=0.6 )
    parser.add_argument('--model_config_path', type=str, default='config/DINO/infer.py', help='root pill image')
    parser.add_argument('--model_checkpoint_path', type=str, default='../weights/checkpoint_best_regular.pth', help='root pill image')

    opt = parser.parse_args()

    main(opt)
