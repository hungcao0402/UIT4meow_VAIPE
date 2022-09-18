import warnings
from PIL import Image
import cv2
from models.builder import MODEL_GETTER
from data.dataset import build_loader
from utils.costom_logger import timeLogger
from utils.config_utils import load_yaml, build_record_folder, get_args
from timm.data import ImageDataset, create_loader, resolve_data_config
import os
import pandas as pd
import torch
import re
import json
import numpy as np
warnings.simplefilter("ignore")
import torchvision.transforms as transforms
from tqdm import tqdm
import shutil

def post_process(filter_id, tmp_score):
    id_map = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    
    top_id = torch.topk(tmp_score[0], 4).indices
    top_id=top_id.tolist()
    top_id = [int(id_map[x]) for x in top_id]
    top_conf = torch.topk(tmp_score[0], 4).values
    top_conf=top_conf.tolist()
                
    class_id = 107
    conf = 1
    # if top_conf[0] <0.1:
    #         return 107, top_conf[0]
    for id, c in zip(top_id,top_conf):
        for box_id in range(len(filter_id)):
            for f_id in filter_id[box_id]:
                if id == f_id:
                    class_id = id
                    conf = c
                    # filter_id[box_id] = [id]
                    # if conf < 0.005:
                    #     return 107, 1
                    return class_id, conf
    return class_id, conf

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

def set_environment(args):
    model = MODEL_GETTER[args.model_name](
        use_fpn=args.use_fpn,
        fpn_size=args.fpn_size,
        use_selection=args.use_selection,
        num_classes=args.num_classes,
        num_selects=args.num_selects,
        use_combiner=args.use_combiner,
    )  # about return_nodes, we use our default setting
    checkpoint = torch.load(args.pretrained, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    model.to(args.device)
    model.eval()

    pres_map_path = os.path.join(args.test_path, 'pill_pres_map.json')
    dataset = DataFromCSV(csv_file=args.csv_path, root_dir=args.input_root, ocr_root=args.ocr_root, pres_map_path=pres_map_path,data_size=args.data_size)

    # dataset = DataCustom(istrain=False, root=args.input_root, root_ocr=args.ocr_root, data_size=args.data_size, return_index=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    print(len(loader))
    return model, loader

def load_map(file):
    f = open(file)
    pill_pres_map = json.load(f)
    new_map = dict()
    for k,v in pill_pres_map.items():
        for i in v:
            new_map[i] = k
    return new_map

def main_test(args, tlogger):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, loader = set_environment(args)
    
    df = pd.read_csv(args.csv_path)
    id_filter = load_ids(args.ocr_root)

    pres_map_path = os.path.join(args.test_path, 'pill_pres_map.json')
    pill_pres_map = load_map(pres_map_path)

    with torch.no_grad():
        count=0
        for batch_idx, (ids,path, images,filter_id) in tqdm(enumerate(loader)):
            filename = os.path.basename(path[0])
            filename_new = filename.split('_',1)[1]
            images = images.to(args.device)
            outs  = model(images)

            tmp_score = torch.softmax(outs["comb_outs"], dim=-1)

            # class_id = tmp_score.argmax().item()
            # class_id = id_map[class_id]
            # conf = tmp_score.max().item()

            path_filter_id = os.path.join(args.ocr_root, pill_pres_map[filename_new]+'.txt')
            ann = open(path_filter_id,'r').read()
            filter_id = json.loads(ann)

            class_id, conf = post_process(filter_id, tmp_score)  
            conf_det = df.loc[df.image_name == filename,'confidence_score'].values[0]
            

            if conf_det < 0.3:
                print(df.loc[df.image_name == filename].index)
                df.drop(df.loc[df.image_name == filename].index, inplace=True)
                continue
                shutil.copy(path[0], os.path.join('temp1/',filename))
                with open('conf_low_1.txt','a') as f:
                    f.write(filename+'\t'+str(conf_det)+'\n')
                # cv2.imwrite('temp/'+)
            # conf_new = conf
            df.loc[df.image_name == filename, ['image_name','class_id','confidence_score']]=[filename_new, class_id, 1.0]
            
    os.makedirs(os.path.dirname(args.output), exist_ok = True) 
    df.to_csv(args.output,index=False)
  


class DataFromCSV(torch.utils.data.Dataset):
    def __init__(self, csv_file, ocr_root, root_dir, pres_map_path, data_size: int):
        self.root_dir = root_dir
        self.ocr_root = ocr_root
        self.data_infos = pd.read_csv(csv_file)
        self.pres_map_path = pres_map_path
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
        self.transforms = transforms.Compose([
                    transforms.Resize((510, 510), Image.BILINEAR),
                    transforms.CenterCrop((data_size, data_size)),
                    transforms.ToTensor(),
                    normalize
            ])
        # self.data_infos = self.getDataInfo(root_dir)
        self.pill_pres_map = self.load_map(pres_map_path)
        self.return_index = True

    def load_map(self, file):
        f = open(file)
        pill_pres_map = json.load(f)
        new_map = dict()
        for k,v in pill_pres_map.items():
            for i in v:
                new_map[i] = k
        return new_map

    def __len__(self):
        
        return len(self.data_infos)

    def __getitem__(self, index):
        img_name = self.data_infos.iloc[index]['image_name']
        new_img_name = img_name.split('_',1)[1]
        image_path = os.path.join(self.root_dir, img_name)
        # label = self.data_infos.iloc[index]['class_id']
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)

        path_filter_id = os.path.join(self.ocr_root, self.pill_pres_map[new_img_name]+'.txt')
        ann = open(path_filter_id,'r').read()
        filter_id = json.loads(ann)

        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, image_path, img, filter_id
        
        # return img, sub_imgs, label, sub_boundarys
        return image_path,img, filter_id



class DataCustom(torch.utils.data.Dataset):
    def __init__(self, 
                 istrain: bool,
                 root: str,
                 root_ocr: str,
                 data_size: int,
                 return_index: bool = False):
        # notice that:
        # sub_data_size mean sub-image's width and height.
        """ basic information """
        self.root = root
        self.data_size = data_size
        self.return_index = return_index
        self.root_ocr= root_ocr
        """ declare data augmentation """
        normalize = transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )

        # 448:600
        # 384:510
        # 768:
        if istrain:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.RandomCrop((data_size, data_size)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        normalize
                ])
        else:
            self.transforms = transforms.Compose([
                        transforms.Resize((510, 510), Image.BILINEAR),
                        transforms.CenterCrop((data_size, data_size)),
                        transforms.ToTensor(),
                        normalize
                ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def getDataInfo(self, root):
        data_infos = []
        files = os.listdir(root)
        for file in files:
            data_path = os.path.join(root, file)
            id_file = re.findall('[0-9]+',file)[0]
            data_infos.append({"path":data_path, "label":1})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, image_path, img,label
        
        # return img, sub_imgs, label, sub_boundarys
        return image_path,img, label

if __name__ == "__main__":
    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    tlogger.print()

    main_test(args, tlogger)
