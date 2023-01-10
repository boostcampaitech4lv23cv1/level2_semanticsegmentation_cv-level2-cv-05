import json
import os
import shutil
from pycocotools.coco import COCO
import cv2
import numpy as np

data_root = '/opt/ml/input/data'

category_names  = ["Backgroud","General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

def copy_img(json_path, json_name ):
    with open(json_path, 'r') as f :
        json_data = json.load(f)
    
    for image in json_data['images']:
        shutil.copyfile(os.path.join(data_root,image['file_name']),
                         os.path.join(data_root,f"mmseg/images/{json_name}/{str(image['id']).zfill(4)}.jpg"))
def gen_mask(json_path, json_name):
    coco = COCO(json_path)

    image_ids = coco.getImgIds()
    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]

        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        # Load the categories in a variable
        cat_ids = coco.getCatIds()
        cats = coco.loadCats(cat_ids)

        # Background = 0
        masks = np.zeros((image_info["height"], image_info["width"]))
        # General trash = 1, ... , Cigarette = 10
        anns = sorted(anns, key=lambda idx : idx['area'], reverse=True)
        for i in range(len(anns)):
            className = get_classname(anns[i]['category_id'], cats)
            pixel_value = category_names.index(className)
            masks[coco.annToMask(anns[i]) == 1] = pixel_value
        masks = masks.astype(np.int8)

        cv2.imwrite(os.path.join(data_root,f"mmseg/annotations/{json_name}/{str(image_info['id']).zfill(4)}.png"), masks)

def main(json_path):
    json_name = json_path.split('/')[-1].split('.')[0]
    
    if json_name != 'test':
        for folder in ['mmseg/images', 'mmseg/annotations']:
            os.makedirs(os.path.join(data_root,os.path.join(folder,json_name)), exist_ok=True)
        copy_img(json_path, json_name)
        gen_mask(json_path, json_name)
    else:
        os.makedirs(os.path.join(data_root,os.path.join('mmseg/images',json_name)), exist_ok=True)
        copy_img(json_path, json_name)


if __name__ =='__main__' :
    main('/opt/ml/input/data/split1_train_MultiStfKFold.json')
    main('/opt/ml/input/data/split1_val_MultiStfKFold.json')
    # main('/opt/ml/input/data/test.json')

