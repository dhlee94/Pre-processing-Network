import argparse

import numpy as np
import torch
import torch.optim as optim

from model import retinanet
from model.dataloader import CocoDataset, collater, BasedSampler
from torch.utils.data import DataLoader
from model import losses
import albumentations
import albumentations.pytorch
from model.utils import AddParserManager, seed_everything

Device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='./config.yml')
parser.add_argument('--config_name', type=str, default='base')


args = parser.parse_args()
cfg = AddParserManager(args.config_file_path, args.config_name)

write_iter_num = cfg.values.write_iter_num
num_epoch = cfg.values.train_args.num_epochs
batch_size = cfg.values.train_args.batch_size
lr = cfg.values.train_args.lr
weight_decay = cfg.values.train_args.weight_decay
network_depth = cfg.values.train_args.network_depth
num_classes = cfg.values.train_args.num_classes
SEED = cfg.values.seed

seed_everything(SEED)

train_transform = albumentations.Compose([
    albumentations.Resize(640, 640),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2()],
    albumentations.BboxParams(format='coco', label_fields=['category_ids']))

val_transform = albumentations.Compose([
    albumentations.Resize(640, 640),
    albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.2, 0.2, 0.2]),
    albumentations.pytorch.transforms.ToTensorV2()],
    albumentations.BboxParams(format='coco', label_fields=['category_ids']))

dataset_train = CocoDataset(cfg.values.coco_path, set_name='train2017', transforms=train_transform)

dataset_val = CocoDataset(cfg.values.coco_path, set_name='val2017', transforms=val_transform)

Sampler = BasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=Sampler)
Sampler_val = BasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
dataloader_val = DataLoader(dataset_val, num_workers=3, batch_sampler=Sampler_val)

if network_depth == 18:
    retinanet = retinanet.resnet18(num_classes=num_classes, pretrained=True)
elif network_depth == 34:
    retinanet = retinanet.resnet34(num_classes=num_classes, pretrained=True)
elif network_depth == 50:
    retinanet = retinanet.resnet50(num_classes=num_classes, pretrained=True)
elif network_depth == 101:
    retinanet = retinanet.resnet101(num_classes=num_classes, pretrained=True)
elif network_depth == 152:
    retinanet = retinanet.resnet152(num_classes=num_classes, pretrained=True)
else:
    raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

model = torch.nn.DataParallel(retinanet).cuda()
model.to(Device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

trainded = False
PATH = "./network_model"
epoch_start = 0

if trainded == True:
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch_start = checkpoint['epoch']
    loss = checkpoint['loss']

for epoch_num in range(epoch_start, num_epoch):
    model.train()
    loss = 0
    epoch_loss = []
    for idx, data in enumerate(dataloader_train):
        imgs = data['img']
        annots = data['annots']
        imgs = imgs.float().to(Device)
        annots = annots.long().to(Device)
        classification_output, regression_output = model((imgs, annots))

        classification_loss = classification_output.mean()
        regression_loss = regression_output.mean()

        loss = classification_loss + regression_loss

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()

        epoch_loss.append(float(loss))

        if idx % write_iter_num == 0:
            print('Epoch : {} || Iteration : {} || Classification loss : {:1.5f} || Regression loss: {:1.5f}'.format(
                epoch_num, idx, float(classification_loss), float(regression_loss)
            ))

        del classification_loss
        del regression_loss

    model.eval()
    coco_eval.evaluate_coco(dataset_val, retinanet)

    scheduler.step(np.mean(epoch_loss))

    torch.save(model, './network_model/{}/model_{}.pt'.format(epoch_num, epoch_num))
    torch.save(model.state_dict(), './network_model/{}/model_{}_model_state_dict.pt',format(epoch_num, epoch_num))
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss},
               './network_model/{}/model_{}_all.tar'.format(epoch_num, epoch_num))

torch.save(model(), 'final_model.pt')
