############################# Import Section #################################

## Imports related to PyTorch
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

## Generic imports
import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

## Dependencies classes and functions
from utils import gridRing
from utils import asMinutes
from utils import timeSince
from utils import getWeights
from utils import videoDatasetRawFrames, videoDatasetPreGenOF
from utils import save_checkpoint
from utils import getListOfFolders

## Import Model
# from DyanC import OFModel
from DyanC import unfrozen_DyanC
############################# Import Section #################################



## HyperParameters for the Network
NumOfPoles = 40
EPOCH = 300
BATCH_SIZE = 1
LR = 0.005
gpu_id = 3
## For training UCF
# Input -  3 Optical Flow
# Output - 1 Optical Flow
## For training Kitti
# Input -  9 Optical Flow
# Output - 1 Optical Flow
FRA = 4 # input number of frame
PRE = 0 # output number of frame
N_FRAME = FRA+PRE
N = NumOfPoles*4
T = FRA # number of row in dictionary(same as input number of frame)
saveEvery = 2


## Load saved model 
load_ckpt = True
# at initialization
trained_encoder = '../preTrainedModel/UCFModel.pth' # for Kitti Dataset: 'KittiModel.pth'
# pick up training
# trained_encoder = checkptname

checkptname = "UCFModel"
ckpt_file = 'UCFModel78.pth'


## Load input data
# set train list name:
trainFolderFile = 'trainlist01.txt'

# set training data directory:
rootDir = '/data/Abhishek/frames/' # RGB
# rootDir = '/storage/truppr/UCF-FLOWS-FULL' # PGOF

trainFoldeList = getListOfFolders(trainFolderFile)[::10]

trainingData = videoDatasetRawFrames(folderList=trainFoldeList, rootDir=rootDir, N_FRAME=N_FRAME) # for raw RGB frames
# trainingData = videoDatasetPreGenOF(folderList=trainFoldeList, rootDir=rootDir, N_FRAME=N_FRAME) # for pre-generated OF

dataloader = DataLoader(trainingData, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

## Initializing r, theta
stateDict = torch.load(trained_encoder)['state_dict']
Dtheta = stateDict['l1.theta']
Drr = stateDict['l1.rr']

## Create the model
model = unfrozen_DyanC(Drr, Dtheta, T, PRE, gpu_id)
# model = DyanC(T, PRE, pretrained_dyan, gpu_id);
model.cuda(gpu_id)
model.train();

'''
stateDict = torch.load(pretrained_dyan)['state_dict'] 
Dtheta = stateDict['l1.theta']
Drr = stateDict['l1.rr']
baseDyan = OFModel(Drr, Dtheta, T, PRE, gpu_id)
baseDyan.eval()
'''

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50,100], gamma=0.1) # if Kitti: milestones=[100,150]

criterion = nn.CrossEntropyLoss()
# loss_mse = nn.MSELoss()
start_epoch = 1

## If want to continue training from a checkpoint
if(load_ckpt):
	loadedcheckpoint = torch.load(ckpt_file)
	start_epoch = loadedcheckpoint['epoch']
	model.load_state_dict(loadedcheckpoint['state_dict'])
	optimizer.load_state_dict(loadedcheckpoint['optimizer'])
	

print("Training from epoch: ", start_epoch)
print('-' * 25)
start = time.time()

## Start the Training
for epoch in range(start_epoch, EPOCH+1):
	loss_value = []
	scheduler.step()
	for i_batch, sample in enumerate(dataloader):
		''' Raw Frames Loading '''
		data = sample['frames'].cuda(gpu_id)
		# expectedOut = Variable(data)
		folderName = sample['ac']
		expectedOut = folderName.cuda(gpu_id);		
		# print('data input size:', data.shape)
		inputData = Variable(data).type(torch.FloatTensor).cuda(gpu_id)
		

		''' PreGen OF Loading '''
		'''
		data = sample['frames'].squeeze(0).cuda(gpu_id)
		# expectedOut = Variable(data)
		folderName = sample['ac']
		expectedOut = folderName.cuda(gpu_id);		
		print('data input size:', data.shape)
		inputData = Variable(data[:,0:FRA,:]).type(torch.cuda.FloatTensor).cuda(gpu_id)
		'''

		optimizer.zero_grad()
	
		# print("\n\/ \/ Network Dimensions \/ \/ ")
		# print(expectedOut.shape)
		# print(expectedOut)

		output = model.forward(inputData.squeeze(0))
		# print('output shape:',output.view(161, 3, 240, 320).shape)
	
		print("\nGround Truth Label -- " + str(expectedOut.item()))
		print("Predicted Label    -- " + str(np.argmax(output.data.cpu().numpy())))
		# print("\nNetwork returned output:\n" + str(output) + "\n")
		# print(output.shape)
		# input();	
		loss = criterion(output, expectedOut)
		loss.backward()
		
		optimizer.step()
		loss_value.append(loss.data.item())

		loss_val = np.mean(np.array(loss_value))

		print('Epoch: ', epoch, '| train loss: %.4f' % loss_val)

		if epoch % saveEvery ==0:
			save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict(),},checkptname+str(epoch)+'.pth')
