import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np 
import os
import re
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
id_filter = load_ids('../output/ocr/exp/labels')

def cluster(box, t):
    feature = np.array([json.loads(box[1])])
    for c in t:
        f2 = np.array([json.loads(c[0][1])])
        score = cosine_similarity(feature, f2)
        if score[0][0] > 0.9:
            c.append(box)
            return t

    t.append([box])

    return t
def vote(top_id,top_conf,filter_id):
    # if int(top_id[0]) == 107:
    #     return 107
    
    for t,c in zip(top_id,top_conf):
        for f_id in filter_id:
            if int(t) in f_id :
                flag=True
                class_id=int(t)
                return int(t),c
    return 107,c

df = pd.read_csv('../output/results.csv')
f = open('/data/public_test/pill_pres_map.json')
datas = json.load(f)
id_map = ['0', '1', '10', '100', '101', '102', '103', '104', '105', '106', '107', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '8', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '9', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']

count = 0
results = []
for data in datas:
    # list_file =[x.replace('json','jpg') for x in data['pill']]    #test_new
    list_file =[x +'.jpg' for x in data['pill']]    #test_new

    temp = pd.DataFrame(columns=['image_name',  'class_id',  'confidence_score','x_min', 'y_min', 'x_max', 'y_max','id'])
    for filename in list_file:
        list1 = df.loc[df.image_name == filename]
        # print(list1)
        temp = pd.concat([temp, list1])
    temp = temp.values.tolist()
    t = []
    for box in temp:
        if not t:
            t.append([box])
            continue
        t = cluster(box, t)
    
    id = 0
    
    id_file = data['pres'].split('.')[0].split('_')[-1]
    
    for i in t:          
        scores = np.array([np.array(json.loads(x[2])) for x in i])
        scores = np.sum(scores, axis=0)
        class_id = id_map[scores.argmax()]
        
        idx = (-scores).argsort()
        top_id = [id_map[x] for x in idx[0][:3]]
        filter_id = id_filter[id_file]

        max_conf = np.max(scores)
        top_conf = scores[0][idx[0][:3]]
        class_id,conf = vote(top_id,top_conf,filter_id)
        print(top_id,top_conf, end=' ')
        print()
        for j in i:
            print(j[0], end=' ')
        print()
        print('------------------------------------')
        for j_index, j in enumerate(i):
            i[j_index] = j[:-1]
            i[j_index][1] = int(class_id)
            i[j_index][2] = conf
    
        id+=1
        results.extend(i)
    break

    

results = pd.DataFrame(results,columns=['image_name',  'class_id',  'confidence_score','x_min', 'y_min', 'x_max', 'y_max'])
results.to_csv('results.csv',index=False)

