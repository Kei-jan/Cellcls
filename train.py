# -*- coding: utf-8 -*-
import os
import sys

import tqdm

import torch
import torch.nn as nn

from torchvision import transforms, datasets, utils
import torch.optim as optim


from sklearn.metrics import classification_report
# from vit_pytorch import SimpleViT,ViT
from vit_pytorch.max_vit import MaxViT
# from vit_pytorch.mobile_vit import MobileViT

import timm
import openpyxl

save_path = '/data/linqx/cellckpt/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

batch_size = 40

epoches = 300

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

model.to(device)

optimizer = optim.Adam(model.parameters(),lr := 1e-04)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, verbose=True)
criterion = nn.CrossEntropyLoss()

data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

train_dataset = datasets.ImageFolder('/data/linqx/celldata/train/',transform=data_transform["train"])

train_num = len(train_dataset)

cell_list = train_dataset.class_to_idx
cla_dict = {val: key for key, val in cell_list.items()}

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size, shuffle=True,num_workers=4)

validate_dataset = datasets.ImageFolder('/data/linqx/celldata/test/',transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,batch_size=batch_size, shuffle=False,num_workers=4)

print(f"using {train_num} images for training, {val_num} images for validation.")

val_best = 0

for e in range(epoches):

    print(f'Epoch: {e} ')

    total_loss = 0
    model.train()

    train_process = tqdm.tqdm(train_loader,file=sys.stdout)
    for step, data in enumerate(train_process):

        optimizer.zero_grad()

        images, labels = data

        outputs = model(images.to(device))

        loss = criterion(outputs,labels.to(device))
        total_loss += loss

        acc = torch.eq(torch.max(outputs,dim=1)[1], labels.to(device)).sum().item()/batch_size

        loss.backward()
        optimizer.step()

        step = step

    tqdm.tqdm.write(f'Epoch: {e}, LR: {lr}\nloss: {total_loss/step}\n')


    with torch.no_grad():

        vls = []
        prds = []
        model.eval()

        eval_loss = 0
        t_acc = 0
        eval_process = tqdm.tqdm(validate_loader,file=sys.stdout)
        for step, data in enumerate(eval_process):

            vimages, vlabels = data
            outputs = model(vimages.to(device))

            vls.append(vlabels)
            prds.append(torch.max(outputs,dim=1)[1])


            loss = criterion(outputs,vlabels.to(device))
            eval_loss += loss

            acc = torch.eq(torch.max(outputs,dim=1)[1], vlabels.to(device)).sum().item()/batch_size

            t_acc += acc

    tqdm.tqdm.write(f"Epoch: {e},\nLoss: {eval_loss/step} ACC: {t_acc/step}\n")

    predfs = torch.cat(prds, 0).detach().cpu().numpy()
    labs = torch.cat(vls, 0)
    confuse_matrix = classification_report(labs, predfs)
    tqdm.tqdm.write(confuse_matrix)

    scheduler.step(eval_loss/step)
    if eval_loss/step < val_best or e%5 == 0:
        val_best = eval_loss/step
        tqdm.tqdm.write("Epoch checkpoint")
        torch.save(model.state_dict(), f'{save_path}/maxvit_4_{e}.pth')
    
