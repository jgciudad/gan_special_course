# -*- coding: utf-8 -*-
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

class FacadeDataset(data.Dataset):
    
    def __init__(self, folder_path): #, CT_size, MR_size):
        super(FacadeDataset, self).__init__()
        # folder_path = folder_path.replace(os.sep, '/')
        self.facade_files = sorted(glob.glob(os.path.join(folder_path,'*.jpg')))
        self.layout_files = sorted(glob.glob(os.path.join(folder_path,'*.png')))
        # self.CT_size = CT_size
      # self.MR_size = MR_size

    def __getitem__(self, index, img_size = (256,256)):
        facade_path = self.facade_files[index]
        layout_path = self.layout_files[index]
        facade_img = Image.open(facade_path).resize(img_size)
        layout_img = Image.open(layout_path).resize(img_size)
        facade_img = np.array(facade_img)
        layout_img = np.array(layout_img)
        # MR_image = MR_image.reshape([MR_image.shape[2],MR_image[0],MR_image[1]])
     
        # if len(image.shape) ==2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # if len(image.shape) > 2 and image.shape[2] == 4:
        #     #if the image is .png (has 4 channels) convert the image from RGBA2RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
        return torch.from_numpy(layout_img).float().unsqueeze(0), torch.from_numpy(facade_img).permute(2,0,1).float(), facade_path
    
    def __len__(self):
        return len(self.facade_files)
    
images_path = '/zhome/02/5/153517/GANS/data/base/'
destination_folder = '/zhome/02/5/153517/GANS/results/pruebaaa/'

facade_dataset = FacadeDataset(images_path)
BATCH_SIZE = 42
train_dataloader = torch.utils.data.DataLoader(facade_dataset, batch_size=BATCH_SIZE)

for i, buildings_data in enumerate(train_dataloader, 0):
	
	layout_images, facade_images, facade_paths = buildings_data
	
	for h in range(BATCH_SIZE):
		fig, ax = plt.subplots(1,2)
		ax[1].imshow(facade_images[h,:,:,:].permute(1,2,0)/255)
		ax[0].imshow(layout_images[h,:,:,:].permute(1,2,0))
		plt.savefig(destination_folder + '/' + str(h) + '.png')
