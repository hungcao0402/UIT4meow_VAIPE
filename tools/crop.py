import os
import glob
import json
import cv2
import tqdm
import random
from pycocotools.coco import COCO

def crop_box():
    root = '/data/public_train/'
    pill_images = root + 'pill/image/*'
    pres_labels = root + 'prescription/image/*'

    pill_label_root = root + 'pill/label/'

    pill_images = glob.glob(pill_images)
    for img_path in tqdm.tqdm(pill_images):
        img = cv2.imread(img_path)

        annot_path = pill_label_root + os.path.basename(img_path).split('.')[0] + '.json'
        with open(annot_path,'r') as f:
            boxes = json.load(f)
        id = 0
        for box in boxes:
            x, y, w, h, label = box['x'], box['y'], box['w'], box['h'], box['label']
            crop_img = img[y:y+h, x:x+w ]
            cv2.imwrite(f'./dataset/classify/train/{label}/' + os.path.basename(img_path).split('.')[0] + '_'+ str(id)+'.jpg', crop_img)
            id+=1

    #crop val
    anno = COCO('./gt.json')
    cats = anno.getAnnIds()
    id = 0
    temp = [0]*108
    for i in cats:
        x = anno.loadAnns(i)[0]
        img_name = x['image_id']
        class_id = x['category_id']
        temp[class_id] +=1
        img_info = anno.loadImgs([img_name])[0]
        w_img, h_img = img_info['width'], img_info['height']

        x,y,w,h = x['bbox']
    c =0
    for x in temp:
        if x > 0:
            c+=1

        img = cv2.imread('/data/public_test/pill/image/'+img_name+'.jpg')
        img_crop = img[y:y+h, x:x+w]
        cv2.imwrite(f'./dataset/val/{class_id}/{id}.jpg',img_crop)

if __name__ == "__main__":
    os.makedirs('./dataset/classify/train', exist_ok = True)
    for i in range(108):
        os.mkdir(f'./dataset/classify/train/{i}')
    os.makedirs('./dataset/classify/val', exist_ok = True)
    for i in range(108):
        os.mkdir(f'./dataset/classify/val/{i}')

    crop_box()
    
    