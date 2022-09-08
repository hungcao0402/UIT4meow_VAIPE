import re
import os
import Levenshtein
import json

def post_process_rec(text): 
    text = text.strip()
    text = text.strip('\"')
    if ')' in text[:7]:
        text = text.split(')',1)[1]

    return text

def process_class_id(rec_result ,path, path_class_pill='./class.txt'):
    file_name = os.path.basename(path)
    id_file = int(re.findall('[0-9]+',file_name)[0])
        
    out_id = []
    with open(path_class_pill, 'r') as f:
        lines = f.readlines()
    
    map = []
    for line in lines:
        id, text = line.split('\t')
        map.append([id,text])
    
    rec_result_new = post_process_rec(rec_result)
 
    for id, key in map:           
        flag = 0
        new_key = post_process_rec(key)
        ratio = Levenshtein.ratio(key.strip(), rec_result_new)
        # print(key, rec_result_new, ratio)
        if ratio > 0.9:
            out_id.append(int(id))
        
    out_id = list(set(out_id))
        
    return out_id

