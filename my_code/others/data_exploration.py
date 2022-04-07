# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 19:38:14 2022

@author: javig
"""
import torch.utils.data as data
import torch
import glob
import os
from PIL import Image
import cv2
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
import SimpleITK as sitk
import matplotlib.pyplot as plt

folder_path = r'C:\Users\javig\OneDrive - Danmarks Tekniske Universitet\Data\Special course GANs\MRXFDG-PET-CT-MRI\MRXFDG-PET-CT-MRI\iDB-CERMEP-MRXFDG_PET_CT_MRI\home\pool\DM\TEP\CERMEP_MXFDG\BASE\DATABASE_SENT\ALL\derivatives\coregistration'
folder_path = folder_path.replace(os.sep, '/')

# example_filename = folder_path + '/sub-0001/' + 'sub-0001_space-pet_ct.nii.gz'
CT_example_filename = folder_path + '/sub-0001/' + 'sub-0001_space-pet_CT.nii.gz'
MR_example_filename = folder_path + '/sub-0001/' + 'sub-0001_space-pet_T1w.nii.gz'
# img = nib.load(example_filename)

# Read the .nii image containing the volume with SimpleITK:
sitk_CT = sitk.ReadImage(CT_example_filename)
sitk_MR = sitk.ReadImage(MR_example_filename)

# # and access the numpy array:
image_CT = sitk.GetArrayFromImage(sitk_CT)
image_CT = np.transpose(image_CT,(2,0,1))
image_CT = np.flip(image_CT,(1))
image_MR = sitk.GetArrayFromImage(sitk_MR)
# image_MR = np.transpose(image_MR,(2,0,1))


#%% Plot CT data
fig, ax = plt.subplots(3,4, figsize = (15,12))

# Along 1st axis
# ax[0,0].imshow(image_CT[40,:,:])
# ax[0,0].set_title('CT 1st axis 50')
# ax[0,1].imshow(image_CT[100,:,:])
# ax[0,1].set_title('CT 1st axis 100')
# ax[0,2].imshow(image_CT[150,:,:])
# ax[0,2].set_title('CT 1st axis 150')
# ax[0,3].imshow(image_CT[200,:,:])
# ax[0,3].set_title('CT 1st axis 250')
ax[0,0].imshow(image_CT[180,:,:])
ax[0,0].set_title('CT 1st axis 50')
ax[0,1].imshow(image_CT[250,:,:])
ax[0,1].set_title('CT 1st axis 100')
ax[0,2].imshow(image_CT[320,:,:])
ax[0,2].set_title('CT 1st axis 150')
ax[0,3].imshow(image_CT[430,:,:])
ax[0,3].set_title('CT 1st axis 250')

# Along 2nd axis
ax[1,0].imshow(image_CT[:,40,:])
ax[1,0].set_title('CT 2nd axis 180')
ax[1,1].imshow(image_CT[:,100,:])
ax[1,1].set_title('CT 2nd axis 250')
ax[1,2].imshow(image_CT[:,150,:])
ax[1,2].set_title('CT 2nd axis 320')
ax[1,3].imshow(image_CT[:,200,:])
ax[1,3].set_title('CT 2nd axis 430')

# Along 3rd axis
ax[2,0].imshow(image_CT[:,:,180])
ax[2,0].set_title('CT 3rd axis 180')
ax[2,1].imshow(image_CT[:,:,250])
ax[2,1].set_title('CT 3rd axis 250')
ax[2,2].imshow(image_CT[:,:,320])
ax[2,2].set_title('CT 3rd axis 320')
ax[2,3].imshow(image_CT[:,:,430])
ax[2,3].set_title('CT 3rd axis 430')


# fig, ax = plt.subplots(1,5)
# plt.imshow(image_MR[80,:,:])