import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from PIL import Image
import matplotlib
import random
from torch.utils.data.sampler import Sampler

class CocoDataset(Dataset):
    def __init__(self, root_dir='./Data', set_name='train2017', split='TRAIN', transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.transform = transforms
        self.whole_image_ids = self.coco.getImgIds()

        self.load_classes()

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
        category_ids = np.array(transform['category_ids']).reshape(-1, 1)

        annots = torch.cat((torch.FloatTensor(bboxes), torch.LongTensor(category_ids)), dim=1)
        sample = {'img' :transform['image'], 'annots':annots}
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.whole_image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        image = Image.open(path).convert('RGB')
        return image, (image_info['width'], image_info['height'])

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def load_annotations(self, image_index):
        annotations_ids = self.coco.getAnnIds(imgIds=self.whole_image_ids[image_index], iscrowd=False)
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

def collater(datas):
    inputs = [data['img'] for data in datas]
    annots = [data['annots'] for data in datas]
    
    widths = [int(s.shape[1]) for s in inputs]
    heights = [int(s.shape[2]) for s in inputs]
    batch_size = len(inputs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    new_imgs = torch.zeros(batch_size, 3, max_width, max_height)

    for i in range(batch_size):
        img = inputs[i]
        new_imgs[i, :, :int(img.shape[1]), :int(img.shape[2])] = img
    
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        new_annots = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    new_annots[idx, :annot.shape[0], :] = annot
    else:
        new_annots = torch.ones((len(annots), 1, 5)) * -1
    

    return {'img' : new_imgs, 'annots' : new_annots}

class BasedSampler(Sampler):
    def __init__(self, data, batch_size, drop_last):
        self.data_source = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
