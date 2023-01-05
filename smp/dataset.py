import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


category_names = [
    'Backgroud',
    'General trash',
    'Paper',
    'Paper pack',
    'Metal',
    'Glass',
    'Plastic',
    'Styrofoam',
    'Plastic bag',
    'Battery',
    'Clothing'
]


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataset(Dataset):
    """COCO format"""
    def __init__(self, dataset_path, json_path, mode='train', transform=None):
        super().__init__()
        self.dataset_path = dataset_path
        self.json_path = json_path
        self.mode = mode
        self.transform = transform
        self.coco = COCO(os.path.join(dataset_path, json_path))

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=index)
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(self.dataset_path,
                                         image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        images /= 255.0

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


class BaseAugmentation:
    def __init__(self, **args):
        self.transform = A.Compose([
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        return self.transform(image=np.array(image), mask=np.int8(mask))


class AugmentationV1:
    def __init__(self, **args):
        self.transform = A.Compose([
            A.ShiftScaleRotate(rotate_method='largest_box', p=0.5),
            A.RandomBrightness(p=0.5),
            ToTensorV2(),
        ])

    def __call__(self, image, mask):
        return self.transform(image=np.array(image), mask=np.int8(mask))


class AugmentationV2:
    def __init__(self, **args):
        self.transform = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ], p=0.5),
            A.OneOf([
                A.ShiftScaleRotate(rotate_method='largest_box', p=1.0),
                A.RandomRotate90(p=0.5),
            ], p=0.8),
            A.RandomResizedCrop(512, 512, scale=(0.8, 1.0), p=0.4),
            A.OneOf([
                A.RandomBrightness(p=1.0),
                A.RandomContrast(p=0.5),
            ], p=0.4),
            A.OneOf([
                A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3,
                             snow_point_upper=0.5, p=0.5),
                A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1,
                               shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1),
                               p=0.5),
                A.RandomRain(brightness_coefficient=0.9, drop_width=1,
                             blur_value=5, p=0.5),
            ], p=0.3),
            ToTensorV2()
        ])

    def __call__(self, image, mask):
        return self.transform(image=np.array(image), mask=np.int8(mask))
