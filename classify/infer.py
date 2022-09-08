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

    dataset = DataCustom(istrain=False, root=args.input_root, root_ocr=args.ocr_root, data_size=args.data_size, return_index=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    return model, loader

def main_test(args, tlogger):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, loader = set_environment(args)
    
    df = pd.read_csv(args.csv_path)
    id_filter = load_ids(args.ocr_root)

    id_map = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106','107', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    out_file = open('write.txt','w')
    with torch.no_grad():
        for batch_idx, (ids,path, images,onehots,labels) in tqdm(enumerate(loader)):
            filename = os.path.basename(path[0])
            filename_new = filename.split('_',1)[1]
            id_file = re.findall('[0-9]+',filename_new)[0]

            images = images.to(args.device)
            onehots = onehots.to(args.device)
            outs  = model(images)
            

            tmp_score = torch.softmax(outs["comb_outs"], dim=-1)
            class_id = tmp_score.argmax().item()
            conf = tmp_score.max().item()
            class_id = int(id_map[class_id])
            filter_id = id_filter[id_file]
            # print(class_id, filter_id)
            
            if class_id not in filter_id:
                class_id=107
            if True:
                top_id = torch.topk(tmp_score[0], 5).indices
                
                filter_id = id_filter[id_file]
                top_id=top_id.tolist()
                top_id = [int(id_map[x]) for x in top_id]
                
                top_conf = torch.topk(tmp_score[0], 5).values
                top_conf=top_conf.tolist()
             
                class_id = 107
                for id, c in zip(top_id, top_conf):
                    if id in filter_id or id == 107:
                        class_id = id
                        conf = c
                        break
            print(class_id)
            # if class_id == 108:
            #     df.drop(df[df.image_name == filename].index, inplace=True)
            #     print('drop')
            #     continue

            if False:
                if conf < 0.1:
                    class_id=107
                    conf = 1
                    # print(class_id, filename)
            conf_det = df.loc[df.image_name == filename,'confidence_score'].values[0]
            # conf_new = 0.5*conf_det + conf*0.5
            conf_new = conf
            # if conf_det < 0.4:
            #     df.drop(df[df.image_name == filename].index, inplace=True)
            #     print(filename)
            #     continue
            out_file.write(f'{filename} {class_id}\n')
            df.loc[df.image_name == filename, ['image_name','class_id','confidence_score']]=[filename_new, class_id, conf_new]
            
            
    df.to_csv(args.output,index=False)
  
         
    
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
                            # 

        self.transforms = transforms.Compose([
                    transforms.Resize((510, 510), Image.BILINEAR),
                    transforms.CenterCrop((data_size, data_size)),
                    transforms.ToTensor(),
                    normalize
            ])

        """ read all data information """
        self.data_infos = self.getDataInfo(root)

    def load_ids(self):
        ids_filter = dict()
        files = os.listdir(self.root_ocr)
        for file in files:
            id_file = re.findall('[0-9]+',file)[0]
            
            path_file = os.path.join(self.root_ocr, file)

            ann = open(path_file,'r').read()

            classes = json.loads(ann)
            
            ids_filter[id_file] = classes
        return ids_filter

    def getDataInfo(self, root):
        data_infos = []
        id_filter = self.load_ids()
        files = os.listdir(root)
        for file in files:
            data_path = os.path.join(root, file)
            id_file = re.findall('[0-9]+',file)[0]
            arr = np.zeros( 108)
            try:
                arr[id_filter[id_file]] = 1
            except:
                arr[-1] = 1

            data_infos.append({"path":data_path,"onehot":arr, "label":1})
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index):
        # get data information.
        image_path = self.data_infos[index]["path"]
        onehot = self.data_infos[index]["onehot"]
        label = self.data_infos[index]["label"]
        # read image by opencv.
        img = cv2.imread(image_path)
        img = img[:, :, ::-1] # BGR to RGB.

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        onehot=torch.Tensor(onehot)
        if self.return_index:
            # return index, img, sub_imgs, label, sub_boundarys
            return index, image_path, img, onehot,label
        
        # return img, sub_imgs, label, sub_boundarys
        return image_path,img, onehot, label

if __name__ == "__main__":
    tlogger = timeLogger()

    tlogger.print("Reading Config...")
    args = get_args()
    assert args.c != "", "Please provide config file (.yaml)"
    load_yaml(args, args.c)
    tlogger.print()

    main_test(args, tlogger)
