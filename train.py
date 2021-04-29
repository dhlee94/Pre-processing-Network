import argparse

import numpy as np
import torch
import torch.optim as optim

from model.mymodel import mymodel
from model.dataloader import CocoDataset
from torch.utils.data import DataLoader
from model import losses
import albumentations
import albumentations.pytorch

Device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='./config.yml')
parser.add_argument('--config_name', type=str, default='base')

args = parser.parse_args()
cfg = AddParserManager(args.config_file_path, args.config_name)

train_transform = albumentations.Compose([
    albumentations.GaussNoise(),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2(),
    albumentations.BboxParams(format='coco', label_fileds=['category_ids'])
])

val_transform = albumentations.Compose([
    albumentations.GaussNoise(),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2(),
    albumentations.BboxParams(format='coco', label_fileds=['category_ids'])
])

dataset_train = CocoDataset(coco_path, set_name='train2017', transforms=train_transform)

dataset_val = CocoDataset(coco_path, set_name='val2018', transforms=val_transform)

dataloader_train = DataLoader(dataset_train, batch_size=3, num_workers=3, shuffle=True, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=3, num_workers=3)

model = mymodel()
model.to(Device)

optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_schedduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

trainded = False
PATH = "./model"
start_epoch = 0

if trainded == True:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    loss = checkpoint['loss']

for epoch_num in range(epoch_start, parser.epochs):
    model.train()
    loss = 0
    epoch_loss = []
    for idx, data in enumerate(dataloader_train):
        classification_output, regression_output = model(data['img'].cuda().float(), annotation['annot'])

        classification_losss = classification_output.mean()
        regression_loss = regression_output.mean()

        loss = classification_loss + regression_loss

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        epoch_loss.append(float(loss))

        print('Epoch : {} || Iteration : {} || Classification loss : {:1.5f} || Regression loss: {:1.5f}'.format(
            epoch_num, idx, float(classification_loss), float(regression_loss)
        ))

        del classification_loss
        del regression_loss

    model.eval()
    coco_eval.evaluate_coco(dataset_val, retinanet)

    scheduler.step(np.mean(epoch_loss))

    torch.save(model, './model/{}/model_{}.pt'.format(epoch_num, epoch_num))
    torch.save(model.state_dict(), './model/{}/model_{}_model_state_dict.pt',format(epoch_num, epoch_num))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss},
               './model/{}/model_{}_all.tar'.format(epoch_num, epoch_num))

torch.save(model(), 'final_model.pt')
