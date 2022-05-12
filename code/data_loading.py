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
import cv2
import numpy as np
# import nibabel as nib
# from torch.utils.data import DataLoader
import SimpleITK as sitk
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


# folder_path = r'C:\Users\javig\Documents\DTU data (not in drive)\GANs data\MRXFDG-PET-CT-MRI\MRXFDG-PET-CT-MRI\iDB-CERMEP-MRXFDG_PET_CT_MRI\home\pool\DM\TEP\CERMEP_MXFDG\BASE\DATABASE_SENT\ALL\derivatives\MNI'
# folder_path = folder_path.replace(os.sep, '/')

# subject = 'sub-0001'
# CT_example_filename = os.path.join(folder_path,subject,subject+'_space-MNI_CT.nii.gz')
# MR_example_filename = os.path.join(folder_path,subject,subject+'_space-MNI_T1w.nii.gz')
# # img = nib.load(CT_example_filename)
# # a = img.get_fdata()

# # Read the .nii image containing the volume with SimpleITK:
# sitk_CT = sitk.ReadImage(CT_example_filename)
# sitk_MR = sitk.ReadImage(MR_example_filename)

# CT_range = (-200,800) # GNRAL WINDOW
# # CT_range = (0,80) # BRAIN WINDOW

# MR_range = (-40,800) # GNRAL WINDOW

# # and access the numpy array:
# image_CT = sitk.GetArrayFromImage(sitk_CT)
# image_MR = sitk.GetArrayFromImage(sitk_MR)
# image_CT_cropped = image_CT[38:219,6:223,18:199]
# image_MR_cropped = image_MR[38:219,6:223,18:199]
# image_CT_cropped = np.flip(image_CT_cropped,(0,1,2))
# image_MR_cropped = np.flip(image_MR_cropped,(0,1,2))

# # WINDOWING PIXEL VALUES
# image_CT_cropped[image_CT_cropped < CT_range[0]] = CT_range[0]
# image_CT_cropped[image_CT_cropped > CT_range[1]] = CT_range[1]
# image_CT_cropped = (image_CT_cropped - CT_range[0]) / np.absolute(CT_range[1] - CT_range[0]) # Rescaling to [0,1]

# image_MR_cropped[image_MR_cropped < MR_range[0]] = MR_range[0]
# image_MR_cropped[image_MR_cropped > MR_range[1]] = MR_range[1]
# image_MR_cropped = (image_MR_cropped - MR_range[0]) / np.absolute(MR_range[1] - MR_range[0]) # Rescaling to [0,1]





# # image_MR = np.transpose(image_MR,(2,0,1))
# f, ax = plt.subplots(2,3,figsize=(10,10))
# ax[0,0].imshow(image_CT_cropped[:,:,120],'gray') 
# ax[0,1].imshow(image_CT_cropped[:,130,:],'gray')
# ax[0,2].imshow(image_CT_cropped[90,:,:],'gray')
# ax[1,0].imshow(image_MR_cropped[:,:,120],'gray')
# ax[1,1].imshow(image_MR_cropped[:,130,:],'gray')
# ax[1,2].imshow(image_MR_cropped[90,:,:],'gray')
# f.suptitle('Before resizing')
# plt.show()

# # f, ax = plt.subplots(2,3,figsize=(10,10))
# # ax[0,0].imshow(cv2.copyMakeBorder(image_CT_cropped[:,:,120],23,24,42,42,40,41,cv2.BORDER_CONSTANT,0),'gray') 
# # ax[0,1].imshow(image_CT_cropped[:,180,:],'gray')
# # ax[0,2].imshow(image_CT_cropped[85,:,:],'gray')
# # ax[1,0].imshow(image_MR_cropped[:,:,120],'gray')
# # ax[1,1].imshow(image_MR_cropped[:,180,:],'gray')
# # ax[1,2].imshow(image_MR_cropped[85,:,:],'gray')
# # f.suptitle('After resizing')
# # plt.show()

# # image_MR_clip = image_MR.copy()
# # image_MR_clip[image_MR_clip<0] = 0
# # plt.imshow(image_MR_clip[150,:,:],'gray')
# # plt.show()

def resample_img(CT_image, MR_image, is_label=False):
    
    # Resample images to 2mm spacing with SimpleITK
    original_spacing = CT_image.GetSpacing()
    out_spacing = MR_image.GetSpacing()
    original_size = CT_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(CT_image.GetDirection())
    resample.SetOutputOrigin(CT_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(CT_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(CT_image)

# Assume to have some sitk image (CT_image) and label (itk_label)
# resampled_sitk_img = resample_img(CT_image, out_spacing=[2.0, 2.0, 2.0], is_label=False)
# resampled_sitk_lbl = resample_img(itk_label, out_spacing=[2.0, 2.0, 2.0], is_label=True)



# the higher the 1st axis, the further I move in longitudinal axis (from the feet to the head) (I will call it X axis)
# the higher the 2nd axis, the further I move in the sagital axis (from the back to the chest) (I will call it Y axis)
# how to know towards where I move in the 3rd axes?

class MRXFDGDataset(data.Dataset):
    
    def __init__(self, folder_path): #, CT_size, MR_size):
        super(MRXFDGDataset, self).__init__()
        folder_path = folder_path.replace(os.sep, '/')
        self.CT_files = glob.glob(os.path.join(folder_path,'*/*pet_ct.nii.gz'))
        self.MR_files = glob.glob(os.path.join(folder_path,'*/*pet_T1w.nii.gz'))
        # self.CT_size = CT_size
      # self.MR_size = MR_size

    def __getitem__(self, index):
        CT_path = self.CT_files[index]
        MR_path = self.MR_files[index]
        sitk_CT = sitk.ReadImage(CT_path)
        sitk_MR = sitk.ReadImage(MR_path)
        CT_image = sitk.GetArrayFromImage(sitk_CT) #.resize(self.img_size))
        MR_image = sitk.GetArrayFromImage(sitk_MR)
        MR_image = MR_image.reshape([MR_image.shape[2],MR_image[0],MR_image[1]])
     
        # if len(image.shape) ==2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # if len(image.shape) > 2 and image.shape[2] == 4:
        #     #if the image is .png (has 4 channels) convert the image from RGBA2RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
        return torch.from_numpy(CT_image).float(), torch.from_numpy(MR_image).float() #.permute(2,1,0)
    
    def __len__(self):
        return len(self.MR_files)
  

class MRXFDGDataset_B(data.Dataset):
    
    def __init__(self, folder_path): #, CT_size, MR_size):
        super(MRXFDGDataset_B, self).__init__()
        folder_path = folder_path.replace(os.sep, '/')
        self.CT_files = glob.glob(os.path.join(folder_path,'*/*_space-MNI_ct.nii.gz'))
        self.MR_files = glob.glob(os.path.join(folder_path,'*/*_space-MNI_T1w.nii.gz'))
        # self.CT_size = CT_size
        # self.MR_size = MR_size
        
        self.CT_range = (-200,400) # GNRAL WINDOW
        # CT_range = (0,80) # BRAIN WINDOW
        self.MR_range = (20,900) # GNRAL WINDOW
        
    def transform(self, input_tensor, target_tensor):
        
        # Random horizontal flipping
        if random.random() > 0.5:
            input_tensor = TF.hflip(input_tensor)
            target_tensor = TF.hflip(target_tensor)
        
        # Normalization
        input_tensor = TF.normalize(input_tensor,(127.5), (127.5))
        target_tensor = TF.normalize(target_tensor,(127.5), (127.5))
        
        return input_tensor, target_tensor
    
    def __getitem__(self, index):
        CT_path = self.CT_files[index]
        MR_path = self.MR_files[index]
        sitk_CT = sitk.ReadImage(CT_path)
        sitk_MR = sitk.ReadImage(MR_path)
        CT_image = sitk.GetArrayFromImage(sitk_CT) #.resize(self.img_size))
        MR_image = sitk.GetArrayFromImage(sitk_MR).astype('float32')
        image_CT_cropped = CT_image[38:219,6:223,18:199]
        image_MR_cropped = MR_image[38:219,6:223,18:199]
        
        # WINDOWING PIXEL VALUES
        image_CT_cropped[image_CT_cropped < self.CT_range[0]] = self.CT_range[0]
        image_CT_cropped[image_CT_cropped > self.CT_range[1]] = self.CT_range[1]
        image_CT_cropped = 255*(image_CT_cropped - self.CT_range[0]) / np.absolute(self.CT_range[1] - self.CT_range[0]) # Rescaling to [0,255]
        
        image_MR_cropped[image_MR_cropped < self.MR_range[0]] = self.MR_range[0]
        image_MR_cropped[image_MR_cropped > self.MR_range[1]] = self.MR_range[1]
        image_MR_cropped = 255*(image_MR_cropped - self.MR_range[0]) / np.absolute(self.MR_range[1] - self.MR_range[0]) # Rescaling to [0,255]

        
        # rand_axis = random.randrange(0, 3, 1)
        rand_axis = 1

        if rand_axis == 0:
            rand_slice = round(random.uniform(0.2, 0.8) * image_CT_cropped.shape[0])
            CT_slice = image_CT_cropped[rand_slice,:,:]
            MR_slice = image_MR_cropped[rand_slice,:,:]
            CT_slice = cv2.copyMakeBorder(CT_slice,19,20,37,38,cv2.BORDER_CONSTANT,0)
            MR_slice = cv2.copyMakeBorder(MR_slice,19,20,37,38,cv2.BORDER_CONSTANT,0)
        elif rand_axis == 1:
            rand_slice = round(random.uniform(0.2, 0.8) * image_CT_cropped.shape[0])
            CT_slice = image_CT_cropped[:,rand_slice,:]
            MR_slice = image_MR_cropped[:,rand_slice,:]
            CT_slice = cv2.copyMakeBorder(CT_slice,37,38,37,38,cv2.BORDER_CONSTANT,0)
            MR_slice = cv2.copyMakeBorder(MR_slice,37,38,37,38,cv2.BORDER_CONSTANT,0)
        elif rand_axis == 2:
            rand_slice = round(random.uniform(0.2, 0.8) * image_CT_cropped.shape[0])
            CT_slice = image_CT_cropped[:,:,rand_slice]
            MR_slice = image_MR_cropped[:,:,rand_slice]
            CT_slice = cv2.copyMakeBorder(CT_slice,37,38,19,20,cv2.BORDER_CONSTANT,0)
            MR_slice = cv2.copyMakeBorder(MR_slice,37,38,19,20,cv2.BORDER_CONSTANT,0)
            
        
        CT_slice = np.flip(CT_slice,(0,1)).copy()
        MR_slice = np.flip(MR_slice,(0,1)).copy()
        # MR_image = MR_image.reshape([MR_image.shape[2],MR_image[0],MR_image[1]])
     
        # if len(image.shape) ==2:
        #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # if len(image.shape) > 2 and image.shape[2] == 4:
        #     #if the image is .png (has 4 channels) convert the image from RGBA2RGB
        #     image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        
        
        CT_tensor = torch.from_numpy(CT_slice).unsqueeze(0).float()
        MR_tensor = torch.from_numpy(MR_slice).unsqueeze(0).float()
        
        # Random flipping and normalization
        CT_tensor, MR_tensor = self.transform(CT_tensor, MR_tensor)
    
        return CT_tensor, MR_tensor, CT_path #.permute(2,1,0)
    
    def __len__(self):
        return len(self.CT_files)
    
# MRXFDG_dataset = MRXFDGDataset_B(folder_path)

# # # Split in train and test
# # LENGTH_TRAIN = np.ceil(len(MRXFDG_dataset)*0.75)
# # LENGTH_TEST = len(MRXFDG_dataset)-LENGTH_TRAIN
# # train_dataset, test_dataset = data.random_split(MRXFDG_dataset, [LENGTH_TRAIN, LENGTH_TEST])

# # Create DataLoaders
# BATCH_SIZE = 12

# train_dataloader = torch.utils.data.DataLoader(MRXFDG_dataset, batch_size=BATCH_SIZE)
# # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=LENGTH_TEST)

# data_iter_train = iter(train_dataloader)
# CT_train, MR_train, CT_paths = data_iter_train.next()

# # data_iter_test = iter(test_dataloader)
# # CT_test, MR_test = data_iter_test.next()

# fig, ax = plt.subplots(4,3,figsize=(10,10))
# for i in range(2):
#     for j in range(3):
#         ax[2*i,j].imshow(CT_train[(i+1)*j,:,:].permute(1,2,0),'gray')
#         ax[2*(i+1)-1,j].imshow(MR_train[(i+1)*j,:,:].permute(1,2,0),'gray')
# plt.show()


# def segment_dataset_and_save(destination_folder, dataloader, model, device):
    
#     for i, buildings_data in enumerate(dataloader, 0):
    
#         facade_images, layout_images, facade_paths = buildings_data
                   
#         output = model(layout_images.to(device))
#         output = TF.normalize(output,(-1),(1/127.5)) # Undo normalization
#         output_numpy = np.transpose(output.cpu().detach().numpy(),(0,2,3,1))
                
#         facade_images = TF.normalize(facade_images,(-1),(1/127.5)) # Undo normalization
#         facade_numpy = np.transpose(facade_images.detach().numpy(),(0,2,3,1))
#         layout_images = TF.normalize(layout_images,(-1),(1/127.5)) # Undo normalization
#         layout_numpy = np.transpose(layout_images.detach().numpy(),(0,2,3,1))
        
        
#         for k, out_img in enumerate(output_numpy):
#            # out_img = (out_img * 255).astype(np.uint8) # undo [0,1] normalization 
#            # out_img = ((out_img + 1) * 127.5).astype(np.uint8) # undo [-1,1] normalization 
#            im = Image.fromarray(out_img.squeeze(2).astype(np.uint8))
           
#            paired_im = np.hstack((layout_numpy[k,:,:,:].squeeze(2),
#                                   255*np.ones((output_numpy[k,:,:,:].shape[0], 50)),
#                                   facade_numpy[k,:,:,:].squeeze(2),
#                                   255*np.ones((output_numpy[k,:,:,:].shape[0], 50)),
#                                   im))
#            paired_im = Image.fromarray(paired_im.astype(np.uint8))
                        
#            im_path = os.path.join(destination_folder,'images',facade_paths[k].split(os.sep)[-2]+'.jpg')
#            paired_im_path = os.path.join(destination_folder,'paired_images',facade_paths[k].split(os.sep)[-2]+'.jpg')
                   
#            if not os.path.exists(os.path.dirname(im_path)):
#                os.makedirs(os.path.dirname(im_path))
#            if not os.path.exists(os.path.dirname(paired_im_path)):
#                os.makedirs(os.path.dirname(paired_im_path))

#            im.save(im_path)
#            paired_im.save(paired_im_path)
           

def generate_and_save_pix2pix(destination_folder, dataloader, model, device):
    
    for i, buildings_data in enumerate(dataloader, 0):
        
        real_CT, real_MR, CT_paths = buildings_data
                   
        fake_CT = model.forward(real_MR.to(device))
        
        fake_CT_numpy = undo_normalization(fake_CT)
        real_CT_numpy = undo_normalization(real_CT)
        real_MR_numpy = undo_normalization(real_MR)
        
        for k, out_img in enumerate(fake_CT_numpy):
            im = Image.fromarray(out_img.squeeze(2).astype(np.uint8))
           
            fig, ax = plt.subplots(1,3,figsize=(20,8))
            ax[0].imshow(real_MR_numpy[k]/255,'gray')
            ax[0].title.set_text('Real MR')
            ax[1].imshow(fake_CT_numpy[k]/255,'gray')
            ax[1].title.set_text('Fake CT')
            ax[2].imshow(real_CT_numpy[k]/255,'gray')
            ax[2].title.set_text('Real CT')
            
            for indv_ax in ax:
                indv_ax.axis('off')
                        
            im_path = os.path.join(destination_folder,'images',CT_paths[k].split(os.sep)[-1][:-4]+'.jpg')
            paired_im_path = os.path.join(destination_folder,'paired_images',CT_paths[k].split(os.sep)[-1][:-4]+'.jpg')
               
            if not os.path.exists(os.path.dirname(im_path)):
                os.makedirs(os.path.dirname(im_path))
            if not os.path.exists(os.path.dirname(paired_im_path)):
                os.makedirs(os.path.dirname(paired_im_path))
            
            im.save(im_path)
            fig.savefig(paired_im_path)
            
            plt.close(fig)
           
           
def undo_normalization(tensor):
    # Un-does [-1,1] normalization and transforms tensor to numpy
    
    if tensor.shape[1] == 1:
        mean = (-1)
        std = (1/127.5)
    elif tensor.shape[1] == 3:
        mean = (-1, -1, -1)
        std = (1/127.5, 1/127.5, 1/127.5)
        
    unnormalized_tensor = TF.normalize(tensor,mean,std) # Undo normalization
    unnormalized_tensor_numpy = np.transpose(unnormalized_tensor.cpu().detach().numpy(),(0,2,3,1))
    
    return unnormalized_tensor_numpy

        
def generate_and_save_cycleGAN(destination_folder, dataloader, model, device):
    
    for i, buildings_data in enumerate(dataloader, 0):
    
        real_CT, real_MR, CT_paths = buildings_data
                   
        fake_CT, rec_MR, fake_MR, rec_CT = model.forward(real_MR.to(device),real_CT.to(device))
        
        fake_CT_numpy = undo_normalization(fake_CT)
        real_CT_numpy = undo_normalization(real_CT)
        real_MR_numpy = undo_normalization(real_MR)
        rec_MR_numpy = undo_normalization(rec_MR)
        fake_MR_numpy = undo_normalization(fake_MR)
        rec_CT_numpy = undo_normalization(rec_CT)
             
        for k, out_img in enumerate(fake_CT_numpy):
            im = Image.fromarray(out_img.squeeze(2).astype(np.uint8))
           
            fig, ax = plt.subplots(3,2,figsize=(10,14))
            ax[0,0].imshow(real_CT_numpy[k]/255,'gray')
            ax[0,0].title.set_text('Real CT')
            ax[1,0].imshow(fake_CT_numpy[k]/255,'gray')
            ax[1,0].title.set_text('Fake CT')
            ax[2,0].imshow(rec_CT_numpy[k]/255,'gray')
            ax[2,0].title.set_text('Rec. CT')
            ax[0,1].imshow(real_MR_numpy[k]/255,'gray')
            ax[0,1].title.set_text('Real MR')
            ax[1,1].imshow(fake_MR_numpy[k]/255,'gray')
            ax[1,1].title.set_text('Fake MR')
            ax[2,1].imshow(rec_MR_numpy[k]/255,'gray')
            ax[2,1].title.set_text('Rec. MR')
            
            for ax_row in ax:
                for ax_elem in ax_row:
                    ax_elem.axis('off')
                         
            im_path = os.path.join(destination_folder,'images',CT_paths[k].split(os.sep)[-1][:-4]+'.jpg')
            paired_im_path = os.path.join(destination_folder,'paired_images',CT_paths[k].split(os.sep)[-1][:-4]+'.jpg')
                    
            if not os.path.exists(os.path.dirname(im_path)):
                os.makedirs(os.path.dirname(im_path))
            if not os.path.exists(os.path.dirname(paired_im_path)):
                os.makedirs(os.path.dirname(paired_im_path))
            
            im.save(im_path)
            fig.savefig(paired_im_path)

            plt.close(fig)
           
           