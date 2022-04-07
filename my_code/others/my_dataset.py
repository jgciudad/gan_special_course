# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:52:43 2022

@author: javig
"""

"""Dataset class template
This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""

from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image


class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    # def modify_commandline_options(parser, is_train):
    #     """Add new dataset-specific options, and rewrite default values for existing options.
    #     Parameters:
    #         parser          -- original option parser
    #         is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
    #     Returns:
    #         the modified parser.
    #     """
    #     parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
    #     parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
    #     return parser

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        # self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        # self.transform = get_transform(opt)
        
        folder_path = folder_path.replace(os.sep, '/')
        self.CT_files = glob.glob(os.path.join(folder_path,'*/*pet_ct.nii.gz'))
        self.MR_files = glob.glob(os.path.join(folder_path,'*/*pet_T1w.nii.gz'))

    # def __getitem__(self, index):
    #     """Return a data point and its metadata information.
    #     Parameters:
    #         index -- a random integer for data indexing
    #     Returns:
    #         a dictionary of data with their names. It usually contains the data itself and its metadata information.
    #     Step 1: get a random image path: e.g., path = self.image_paths[index]
    #     Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
    #     Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
    #     Step 4: return a data point as a dictionary.
    #     """
    #     path = 'temp'    # needs to be a string
    #     data_A = None    # needs to be a tensor
    #     data_B = None    # needs to be a tensor
    #     return {'data_A': data_A, 'data_B': data_B, 'path': path}
    
    
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
        return {'data_CT': torch.from_numpy(CT_image).float(), 'data_B': data_B, 'path': path}
    
    def __len__(self):
        return len(self.MR_files)

    # def __len__(self):
    #     """Return the total number of images."""
    #     return len(self.image_paths)