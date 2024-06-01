# -*- coding: utf-8 -*-
import os
import sys

from tqdm import tqdm
import json
import torch
import torch.nn as nn

from torchvision import transforms, datasets, utils
import torch.optim as optim

from vit_pytorch import SimpleViT,ViT
from vit_pytorch.max_vit import MaxViT
from vit_pytorch.mobile_vit import MobileViT

import timm

from sklearn.metrics import classification_report
# from dataload.dataloader import celloader
import openpyxl

save_path = '/data/linqx/cellckpt/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

batch_size = 1

model = MaxViT(
    num_classes = 7,
    dim_conv_stem = 64,               # dimension of the convolutional stem, would default to dimension of first layer if not specified
    dim = 192,                         # 96 dimension of first layer, doubles every layer
    dim_head = 32,                    # dimension of attention heads, kept at 32 in paper
    depth = (2, 2, 5, 2),             # number of MaxViT blocks per stage, which consists of MBConv, block-like attention, grid-like attention
    window_size = 7,                  # window size for block and grids
    mbconv_expansion_rate = 4,        # expansion rate of MBConv
    mbconv_shrinkage_rate = 0.25,     # shrinkage rate of squeeze-excitation in MBConv
    dropout = 0.1                     # dropout
)

ckpt = torch.load(f'{save_path}/maxvit_2_0.pth')
model.load_state_dict(ckpt)
model.eval()
model.to(device)

data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}


test_dataset = datasets.ImageFolder('/data/linqx/celldata/test/',transform=data_transform["val"])


test_num = len(test_dataset)


flower_list = test_dataset.class_to_idx
cla_dict = {val: key for key, val in flower_list.items()}

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size, shuffle=False,num_workers=4)

tqdm.write(f"{test_num} images.")


wb = openpyxl.Workbook()
wb.create_sheet('last_features')
ws = wb['last_features']
labels = []
preds = []
predfs = []
xls_path = './result.xlsx'
t_acc = 0
with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for step, data in enumerate(test_bar):
        images, label = data
        images = images.to(device)
        labels.append(label)
        pred_feature, pred = model(images)
        # tqdm.write(f'{pred_feature.shape}')
        preds.append(pred)
        predf = torch.max(pred,dim=1)[1]
        # tqdm.write(f'{predf}')
        # tqdm.write(f'{label}')
        acc = torch.eq(torch.max(pred,dim=1)[1], label.to(device)).sum().item()

        t_acc += acc
        # tqdm.write(f'{acc}')
        predfs.append(predf)
        ws.append(pred_feature.detach().cpu().numpy().tolist()[0])
    
    
    preds = torch.cat(preds, 0).detach().cpu().numpy()
    predfs = torch.cat(predfs, 0).detach().cpu().numpy()
    labels = torch.cat(labels, 0)
    confuse_matrix = classification_report(labels, predfs)
    tqdm.write(f'{t_acc/step}')
    tqdm.write(confuse_matrix)

    classification_dict = dict()
    classification_dict['prediction'] = preds.tolist()
    classification_dict['ground truth'] = labels.numpy().tolist()
    json_path = './result.json'
    with open(os.path.join(json_path), 'w') as json_file:
        # writing the dictionary data into the corresponding JSON file
        json.dump(classification_dict, json_file, indent=5)

    txt_path = './result.txt'
    txt = open(txt_path, 'w')
    txt.write(confuse_matrix)
    # wb.save(xls_path)
