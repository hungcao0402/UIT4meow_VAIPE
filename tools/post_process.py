import csv
import pandas as pd
import json
import os
import argparse

def load_map(file):
    f = open(file)
    pill_pres_map = json.load(f)
    new_map = dict()
    for k,v in pill_pres_map.items():
        for i in v:
            new_map[i.lower()] = k
    return new_map                                                                                           

def convert_to_coco_format(csv_path):
    pred_results = []
    df = pd.read_csv(csv_path)
    outputs = []
    for data in df.values:
        img_name = data[0]
        image_id = img_name.split('.')[0]
        category_id = int(data[1])
        conf = data[2]
        xmin, ymin, xmax,ymax = data[3:]
        
        bboxes = [xmin, ymin, xmax, ymax]

        bbox = [round(x, 3) for x in bboxes]

        score = round(conf, 17)
        # score = round(conf, 16)
        pred_data = {
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score
        }

        pred_results.append(pred_data)
        
    return pred_results


def in_dict(image_id, annotations):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            return True
    return False

def merge_dict(annotations):
    merged_annotations = []
    for i in range(len(annotations)):
        image_id = annotations[i]['image_id']
        annos = []
        if in_dict(image_id, merged_annotations) == False:
            for j in range(len(annotations)):
                if image_id == annotations[j]['image_id']:
                    annos.append({'category_id': annotations[j]['category_id'], 'bbox': annotations[j]['bbox'], 'score': annotations[j]['score']})
            merged_annotations.append({'image_id': image_id, 'annotations': annos})
    return merged_annotations

def get_annotation(image_id, annotations):
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            return annotation['annotations']
    print(image_id)

def expand_size(bbox, percent):
    x_min = float(bbox[0])
    y_min = float(bbox[1])
    x_max = float(bbox[2])
    y_max = float(bbox[3])
    width = x_max - x_min
    height = y_max - y_min
    x_min = x_min - width * (percent / 2)
    y_min = y_min - height * (percent / 2)
    x_max = x_max + width * (percent / 2)
    y_max = y_max + height * (percent / 2)
    return [x_min, y_min, x_max, y_max]


def write_results(results, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max'])
        for result in results:
            image_id = result['image_id']
            for ann in result['annotations']:
                bboxes = ann['bbox']
                score = ann['score']
                label = ann['category_id']
                xmin = float(bboxes[0])
                ymin = float(bboxes[1])
                xmax = float(bboxes[2])
                ymax = float(bboxes[3])
                writer.writerow([image_id + '.jpg', str(label), score, xmin, ymin, xmax, ymax])


def post_process_one_bbox(csv_path, output_path, opt):
    results = convert_to_coco_format(csv_path)
    results = merge_dict(results)
    pres_map_path = os.path.join(opt.test_path, "pill_pres_map.json")
    pill_pres_map = load_map(pres_map_path)
    # js = json.dumps(pill_pres_map)
    # with open('test.json', 'w') as f:
    #     f.write(js)

    for result in results:
        image_id = result['image_id']
        annotations = result['annotations']

        for i in range(len(result['annotations'])):
            result['annotations'][i]['score'] = 1.0

        if(len(annotations) == 1):
            path_filter_id = os.path.join(opt.ocr_root, pill_pres_map[image_id.lower()+'.jpg']+'.txt')
            ann = open(path_filter_id,'r').read()
            filter_id = json.loads(ann)
            temp = []
            for i in filter_id:
                temp += i
            temp = list(set(temp))
            # map to filter_id
            if len(temp) == 1:
                result['annotations'][0]['category_id'] = temp[0]
    
    write_results(results, output_path)

parser = argparse.ArgumentParser()
parser.add_argument('--test_path',  type=str, default='/data/private_test', help='')
parser.add_argument('--ocr_root',  type=str, default='/temp/ocr/exp/labels', help='')

opt = parser.parse_args()

post_process_one_bbox('./output/results.csv','./output/results.csv', opt)