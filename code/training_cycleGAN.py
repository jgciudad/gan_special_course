# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 13:12:53 2022

@author: javig
"""

'''
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
'''

import time
import os
from data_loading_facades import *
from my_cycleGAN import *
import datetime
import matplotlib.pyplot as plt


# -------------------------------------- PATHS ------------------------------------------------------------

# test_path = '/zhome/02/5/153517/GANS/results/test_37/'
# images_path_train = '/zhome/02/5/153517/GANS/data/base2/train/'
# images_path_test = '/zhome/02/5/153517/GANS/data/base2/test/'

test_path = r'C:\Users\javig\Desktop\eliminar_esta_carpeta/'
test_path = test_path.replace(os.sep,'/')
images_path_train = r'C:\Users\javig\Documents\DTU data (not in drive)\GANs data\CMP_facade_DB_base\base/train/'
images_path_train = images_path_train.replace(os.sep, '/')
images_path_test = r'C:\Users\javig\Documents\DTU data (not in drive)\GANs data\CMP_facade_DB_base\base/test/'
images_path_test = images_path_test.replace(os.sep, '/')

#-------------------------------------------------------------------------------------------------------------

program_start_time = time.time()

facade_dataset_train = FacadeDataset(images_path_train)
facade_dataset_test = FacadeDataset(images_path_test)

model = CycleGANModel(input_nc = 1,
                      output_nc = 3)

if not os.path.exists(os.path.dirname(test_path)):
    os.makedirs(os.path.dirname(test_path))
    
NUM_EPOCHS = 1
D_STARTING_EPOCH = 0
BATCH_SIZE = 1

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(facade_dataset_train, batch_size=BATCH_SIZE)
test_dataloader = torch.utils.data.DataLoader(facade_dataset_test, batch_size=len(facade_dataset_test))

data_iter_train = iter(train_dataloader)
layout_train, facade_train, facade_paths_train = data_iter_train.next()
data_iter_test = iter(test_dataloader)
layout_test, facade_test, facade_paths_test = data_iter_test.next()

D_A_history_epochs = []
D_B_history_epochs = []
G_A_history_epochs = []
G_B_history_epochs = []
cycle_A_history_epochs = []
cycle_B_history_epochs = []
G_history_epochs = []


for epoch in range(NUM_EPOCHS):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
           
    print('Epoch', epoch)

    epoch_start_time = time.time()  # timer for entire epoch
    
    D_A_history_batches = []
    D_B_history_batches = []
    G_A_history_batches = []
    G_B_history_batches = []
    cycle_A_history_batches = []
    cycle_B_history_batches = []
    G_history_batches = []

    for i, data in enumerate(train_dataloader):  # inner loop within one epoch

        
        real_A, real_B, facade_paths = data
        
        real_A, real_B = real_A.to(model.device), real_B.to(model.device)

        model.optimize_parameters(real_A, real_B)   # calculate loss functions, get gradients, update network weights
        
        losses = model.get_current_losses()
        
        D_A_history_batches.append(losses['D_A'])
        D_B_history_batches.append(losses['D_B'])
        G_A_history_batches.append(losses['G_A'])
        G_B_history_batches.append(losses['G_B'])
        cycle_A_history_batches.append(losses['cycle_A'])
        cycle_B_history_batches.append(losses['cycle_B'])
        G_history_batches.append(losses['G'])

    
    D_A_history_epochs.append(np.mean(D_A_history_batches))
    D_B_history_epochs.append(np.mean(D_B_history_batches))
    G_A_history_epochs.append(np.mean(G_A_history_batches))
    G_B_history_epochs.append(np.mean(G_B_history_batches))
    cycle_A_history_epochs.append(np.mean(cycle_A_history_batches))
    cycle_B_history_epochs.append(np.mean(cycle_B_history_batches))  
    G_history_epochs.append(np.mean(G_history_batches))  
    
    if (epoch+1) in [1,60,100,150,200,250,300,350,400]:
        destination_epoch_train = os.path.join(test_path,'generated_images','train','epoch_'+str(epoch+1))
        segment_dataset_and_save(destination_epoch_train, train_dataloader, model.netG_A, model.device)
        destination_epoch_test = os.path.join(test_path,'generated_images','test','epoch_'+str(epoch+1))
        segment_dataset_and_save(destination_epoch_test, test_dataloader, model.netG_A, model.device)
        
    print('Time taken:', str(datetime.timedelta(seconds = time.time()-epoch_start_time)))

fig1 = plt.figure()
plt.plot(D_A_history_epochs)
plt.plot(D_B_history_epochs)
plt.xlabel('Epochs')
plt.ylabel('BCE')
plt.legend(['D_A','D_B'])
plt.title('D_loss (Loss of D)')
plt.savefig(test_path + '/D_loss_training_plot.png')

fig2 = plt.figure()
plt.plot(G_A_history_epochs)
plt.plot(G_B_history_epochs)
plt.xlabel('Epochs')
plt.ylabel('BCE')
plt.legend(['G_A','G_B'])
plt.title('G loss')
plt.savefig(test_path + '/D_fake_training_plot.png')

fig3 = plt.figure()
plt.plot(cycle_A_history_epochs)
plt.plot(cycle_B_history_epochs)
plt.xlabel('Epochs')
plt.ylabel('BCE')
plt.legend(['cycle_A','cycle_B'])
plt.title('cycle loss')
plt.savefig(test_path + '/D_real_training_plot.png')

fig4 = plt.figure()
plt.plot(G_history_epochs)
plt.xlabel('Epochs')
plt.ylabel('BCE + L1')
plt.legend(['Train','Test'])
# plt.title('G loss (sum of G_GAN and G_L1)')
plt.savefig(test_path + '/G_training_plot.png')

# fig5 = plt.figure()
# plt.plot(G_GAN_history_epochs)
# # plt.plot(loss_history_test_epochs)
# plt.xlabel('Epochs')
# plt.ylabel('BCE')
# plt.legend(['Train','Test'])
# plt.title('G_GAN (loss of D for classifying fake images as true)')
# plt.savefig(test_path + '/G_GAN_training_plot.png')

# fig6 = plt.figure()
# plt.plot(G_L1_history_epochs)
# # plt.plot(IoU_history_test_epochs)
# plt.xlabel('Epochs')
# plt.ylabel('L1')
# plt.legend(['Train','Test'])
# plt.title('G_L1 (loss of G for generating an image similar to the true image)')
# plt.savefig(test_path + '/G_L1_training_plot.png')


print('Total time taken:', str(datetime.timedelta(seconds = time.time()-program_start_time)))