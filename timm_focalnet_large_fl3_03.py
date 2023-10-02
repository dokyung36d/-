import os
import cv2
from PIL import Image
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modelNetM_03 import EncoderNet, DecoderNet, ClassNet, EPELoss, make_tensor
from urllib.request import urlopen
import timm
from torchvision.transforms import ToPILImage

def run():
    torch.multiprocessing.freeze_support()



class CustomDataset_for_train(Dataset):
    def __init__(self, train, val, transform=None, infer=False):

        self.train = pd.read_csv(train)
        self.val = pd.read_csv(val)
        self.data = pd.concat([self.train, self.val])
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.data.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask
    
class CustomDataset_for_test(Dataset):
    def __init__(self, csv_file, transform=None, infer=False):

        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.data.iloc[idx, 2]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    
if __name__ == '__main__':
    run()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)

    # RLE 인코딩 함수
    def rle_encode(mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

        
    transform = A.Compose(
        [   
            A.Resize(224, 224),
            A.Rotate(limit=3, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(),
            ToTensorV2()
        ]
    )

    dataset = CustomDataset_for_train(train='./train_source.csv',val = './val_source.csv', transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    model_en = EncoderNet([1,1,1,1,2])
    model_de = DecoderNet([1,1,1,1,2])
    model_class = ClassNet()
    criterion = EPELoss()

    model = timm.create_model(
        'focalnet_large_fl3.ms_in22k',
        pretrained=True,
        features_only=True,
    )  # remove classifier nn.Linear)
    criterion_clas = nn.CrossEntropyLoss()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_en = nn.DataParallel(model_en)
        model_de = nn.DataParallel(model_de)
        model_class = nn.DataParallel(model_class)

    if torch.cuda.is_available():
        model = model.cuda()
        model_en = model_en.cuda()
        model_de = model_de.cuda()
        model_class = model_class.cuda()
        criterion = criterion.cuda()
        criterion_clas = criterion_clas.cuda()

    optimizer = torch.optim.Adam(list(model_en.parameters()) + list(model_de.parameters()))

    #model.train()  --> 사용할 수도 있음
    model_en.train()
    model_de.train()
    model_class.train()

    to_pil = ToPILImage()

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # training loop

    num_of_epochs = 50
    for epoch in range(num_of_epochs):
        epoch_loss = 0
        for images, masks in tqdm(dataloader):
            # print("masks.shape : ", masks.shape)
            images = images.float().to(device)
            masks = masks.long().to(device)

            new_images = torch.zeros(images.shape[0], 3, 384, 384)
            images = images.to('cpu')      
            for i in range(len(images)):
                for j in range(len(images[i])):
                    new_images[i][j] = torch.tensor(cv2.resize(np.array(images[i][j]), (384, 384)))
            images = new_images.float().to(device)

            optimizer.zero_grad()
            # print(images.shape)


            first_output_list = []
            for image in images:
                image = image.to(device)
                internal_out = torch.tensor([]).to(device)
                first_output = model(transforms(to_pil(image)).unsqueeze(0).to(device))
                first_output_list.append(first_output)


            output = model_en(make_tensor([row[0] for row in first_output_list]).to(device)
                              , make_tensor([row[1] for row in first_output_list]).to(device)
                              , make_tensor([row[2] for row in first_output_list]).to(device)
                              , make_tensor([row[3] for row in first_output_list]).to(device))
                # outputs = torch.cat((outputs, model(transforms(to_pil(image)).unsqueeze(0).cuda())), dim=0)

            # output = torch.tensor([])
            # for i in range(len(outputs)):
            #     output = torch.cat((output, outputs[i]), dim = 0)
            ##output.shape : (batch, 1024)

            # middle = model_en(output) # shape of output: 
            # print(middle.shape)
            # # clas = model_class(middle)
            # # print("clas shape ; ", clas.shape)

            flow_output = model_de(output)
            # print("flow outpus shape : ",flow_output.shape)


            loss = criterion_clas(flow_output, masks.squeeze(1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # print("no error occured!!!!!!")



        print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}')


        torch.save(model_en, f"model_en_03_{epoch+1}for30.pt")
        torch.save(model_de, f"model_de_03_{epoch+1}for30.pt")


    # parameter save

    print("training process terminated")


    test_dataset = CustomDataset_for_test(csv_file='./test.csv', transform=transform, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    with torch.no_grad():
        model.eval()
        model_en.eval()
        model_de.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)
            new_images = torch.zeros(images.shape[0], 3, 384, 384)
            images.to('cpu')
            for i in range(len(images)):
                for j in range(len(images[i])):
                    new_images[i][j] = torch.tensor(cv2.resize(np.array(images[i][j].to('cpu')), (384, 384)))
            images = new_images.float().to(device)

            optimizer.zero_grad()
            # print(images.shape)


            first_output_list = []
            for image in images:
                image = image.to(device)
                internal_out = torch.tensor([]).to(device)
                first_output = model(transforms(to_pil(image)).unsqueeze(0).to(device))
                first_output_list.append(first_output)


            output = model_en(make_tensor([row[0] for row in first_output_list]).to(device)
                              , make_tensor([row[1] for row in first_output_list]).to(device)
                              , make_tensor([row[2] for row in first_output_list]).to(device)
                              , make_tensor([row[3] for row in first_output_list]).to(device))
            outputs = model_de(output)

            outputs = torch.softmax(outputs, dim=1).cpu()
            outputs = torch.argmax(outputs, dim=1).numpy()
            # batch에 존재하는 각 이미지에 대해서 반복
            for pred in outputs:
                pred = pred.astype(np.uint8)
                pred = Image.fromarray(pred) # 이미지로 변환
                pred = pred.resize((960, 540), Image.NEAREST) # 960 x 540 사이즈로 변환
                pred = np.array(pred) # 다시 수치로 변환
                # class 0 ~ 11에 해당하는 경우에 마스크 형성 / 12(배경)는 제외하고 진행
                for class_id in range(12):
                    class_mask = (pred == class_id).astype(np.uint8)
                    if np.sum(class_mask) > 0: # 마스크가 존재하는 경우 encode
                        mask_rle = rle_encode(class_mask)
                        result.append(mask_rle)
                    else: # 마스크가 존재하지 않는 경우 -1
                        result.append(-1)


    submit = pd.read_csv('./sample_submission.csv')
    submit['mask_rle'] = result
    submit

    submit.to_csv(f'./baseline_submit_03_{num_of_epochs}.csv', index=False)