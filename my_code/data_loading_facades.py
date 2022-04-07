# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 19:15:31 2022

@author: javig
"""
import torch.utils.data as data
import torch
import glob
import os
from PIL import Image
# import cv2
import numpy as np
# import nibabel as nib
# from torch.utils.data import DataLoader
# import SimpleITK as sitk
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random



folder_path = r'C:\Users\javig\Documents\DTU data (not in drive)\GANs data\CMP_facade_DB_base\base/'
folder_path = folder_path.replace(os.sep, '/')

# facade_example_filename = folder_path + 'cmp_b0001.jpg'
# layout_example_filename = folder_path + 'cmp_b0001.png'

# Read the .nii image containing the volume with SimpleITK:
# face_img = Image.open(facade_example_filename)
# layout_img = Image.open(layout_example_filename)

# face_img.show()
# layout_img.show()

class FacadeDataset(data.Dataset):
    
    def __init__(self, folder_path): #, CT_size, MR_size):
        super(FacadeDataset, self).__init__()
        # folder_path = folder_path.replace(os.sep, '/')
        self.facade_files = sorted(glob.glob(os.path.join(folder_path,'*.jpg')))
        self.layout_files = sorted(glob.glob(os.path.join(folder_path,'*.png')))
        # self.trans = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
        #                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.CT_size = CT_size
      # self.MR_size = MR_size
      
    def transform(self, input_tensor, target_tensor):
        # Random horizontal flipping
        if random.random() > 0.5:
            input_tensor = TF.hflip(input_tensor)
            target_tensor = TF.hflip(target_tensor)
            
        input_tensor = TF.normalize(input_tensor,(0.5), (0.5))
        target_tensor = TF.normalize(target_tensor,(0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return input_tensor, target_tensor

    def __getitem__(self, index, img_size = (256,256)):
        facade_path = self.facade_files[index]
        layout_path = self.layout_files[index]
        facade_img = Image.open(facade_path).resize(img_size)
        layout_img = Image.open(layout_path).resize(img_size)
        facade_img = np.array(facade_img)/255
        layout_img = np.array(layout_img)/255
        # MR_image = MR_image.reshape([MR_image.shape[2],MR_image[0],MR_image[1]])
     
        # if len(image.shape) ==2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # if len(image.shape) > 2 and image.shape[2] == 4:
        #     #if the image is .png (has 4 channels) convert the image from RGBA2RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        layout_tensor = torch.from_numpy(layout_img).float().unsqueeze(0)
        facade_tensor =  torch.from_numpy(facade_img).permute(2,0,1).float()
        
        layout_tensor, facade_tensor = self.transform(layout_tensor, facade_tensor)
    
        return layout_tensor, facade_tensor, facade_path
    
    def __len__(self):
        return len(self.facade_files)


def segment_dataset_and_save(destination_folder, dataloader, model, device):
    
    for i, buildings_data in enumerate(dataloader, 0):
    
        layout_images, facade_images, facade_paths = buildings_data
                          
        output = model(layout_images.to(device))
        output = TF.normalize(output,(-1, -1, -1),(1/0.5, 1/0.5, 1/0.5))
        output_numpy = np.transpose(output.cpu().detach().numpy(),(0,2,3,1))
        # output_numpy = output_numpy*255
        
        facade_numpy = np.transpose(facade_images.detach().numpy(),(0,2,3,1))
        facade_numpy = facade_numpy * 255
        
        for k, out_img in enumerate(output_numpy):
           out_img = (out_img * 255).astype(np.uint8)
           im = Image.fromarray(out_img)
           
           paired_im = np.hstack((facade_numpy[k,:,:,:], 255*np.ones((output_numpy[k,:,:,:].shape[0], 50, 3)), im))
           paired_im = Image.fromarray(paired_im.astype(np.uint8))
                        
           im_path = os.path.join(destination_folder,'images',facade_paths[k].split(os.sep)[-1][:-4]+'.jpg')
           # im_path = im_path.replace(os.sep,'/')
           
           paired_im_path = os.path.join(destination_folder,'paired_images',facade_paths[k].split(os.sep)[-1][:-4]+'.jpg')
           # paired_im_path = paired_im_path.replace(os.sep,'/')
                   
           if not os.path.exists(os.path.dirname(im_path)):
               os.makedirs(os.path.dirname(im_path))
           if not os.path.exists(os.path.dirname(paired_im_path)):
               os.makedirs(os.path.dirname(paired_im_path))

           im.save(im_path)
           paired_im.save(paired_im_path)

  
# Facade_data = FacadeDataset(folder_path)

# # Split in train and test
# LENGTH_TRAIN = np.ceil(len(MRXFDG_dataset)*0.75)
# LENGTH_TEST = len(MRXFDG_dataset)-LENGTH_TRAIN
# train_dataset, test_dataset = data.random_split(MRXFDG_dataset, [LENGTH_TRAIN, LENGTH_TEST])

# Create DataLoaders
# BATCH_SIZE = 12

# train_dataloader = torch.utils.data.DataLoader(Facade_data, batch_size=BATCH_SIZE)
# test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=LENGTH_TEST)

# data_iter_train = iter(train_dataloader)
# layout_train, facade_train, paths = data_iter_train.next()

# data_iter_test = iter(test_dataloader)
# CT_test, MR_test = data_iter_test.next()

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# images_test, labels_test = images_test.to(device), labels_test.to(device)
# images_test, labels_test = Variable(images_test), Variable(labels_test)

# for h in range(12):
#     fig, ax = plt.subplots(1,2)
#     ax[1].imshow(facade_train[h,:,:,:].permute(1,2,0)/255)
#     ax[0].imshow(layout_train[h,:,:,:].permute(1,2,0))

