import tqdm
import os
import json
import cv2 

def pres(root):
    os.makedirs('./dataset/pres/vietocr/img', exist_ok = True)
    id = 0 
    pres_labels_path = root + 'prescription/label'
    train_file = open('./dataset/pres/vietocr/train.txt','w', encoding='utf-8')
    val_file = open('./dataset/pres/vietocr/val.txt','w', encoding='utf-8')
    len_train = len(os.listdir(pres_labels_path))*0.8

    count = 0
    for file in tqdm.tqdm(os.listdir(pres_labels_path)):
        file_path = os.path.join(pres_labels_path,file)
        with open(file_path,'r') as f:
            boxes = json.load(f)
        img_path = file_path.replace('label','image').replace('json','png')
        img = cv2.imread(img_path)
        
        for box in boxes:
            label = box['label']
            if label == 'drugname':
                id+=1
                text =   box['text']
                x,y,x1,y1 = box['box']
                cv2.imwrite(f'./dataset/pres/vietocr/img/{id:08}.jpg', img[y:y1,x:x1])
                if count < len_train:
                    train_file.write(f'img/{id:08}.jpg\t{text}\n')
                else:
                    val_file.write(f'img/{id:08}.jpg\t{text}\n')
        count+=1
root = '/data/public_train/'

pres(root)