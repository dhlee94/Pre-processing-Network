import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib

class CocoDataset(Dataset):
    def __init__(self, root_dir='./Data', set_name='train2017', split='TRAIN', transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.transform = transforms
        self.whole_image_ids = self.coco.getImgIds()

        self.load_classes()
        self.image_ids = []
        self.no_anno_list = []

        for idx in self.whole_image_ids:
            annotations_ids = self.coco.getAnnIds(imgIds=idx, iscrowd=False)
            if len(annotations_ids) == 0:
                self.no_anno_list.append(idx)
            else:
                self.image_ids.append(idx)

    def __len__(self):
        return len(self.whole_image_ids)

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])
         
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}

        for c in categories:
           self.coco_labels[len(self.classes)] = c['id']
           self.coco_labels_inverse[c['id']] = len(self.classes)
           self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
           self.labels[value] = key

    def __getitem__(self, idx):

        image, (w, h) = self.load_image(idx)

        annotation = self.load_annotations(idx)

        boxes = torch.FloatTensor(annotation[:, :4])
        labels = torch.LongTensor(annotation[:,4])
        if self.transform:
            transform = self.transform(image=np.array(image), bboxes=boxes, category_ids=labels)
        bboxes = transform['bboxes']
        category_ids = transform['category_ids'].unsqueeze(1)

        annots = torch.cat((bboxes, category_ids), dim=1)
        sample = {'img' :transform['image'], 'annots':annots}
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image, (image_info['width'], image_info['height'])

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        if len(annotations_ids) == 0:
            return annotations

        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

#def collater(datas):
#    inputs = [data['img'] for data in datas]
#    labels = [data['label'] for data in datas]
#    
#    widths = [int(s.shape[0]) for s in inputs]
#    heights = [int(s.shape[1]) for s in inputs]
#    batch_size =
