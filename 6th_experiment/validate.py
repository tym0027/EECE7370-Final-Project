############################# Import Section #################################
## Imports related to PyTorch
import torch
from torch.autograd import Variable

## Generic imports
import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize

# from DyanOF import OFModel
from DyanC import OFModel
from DyanC import unfrozen_DyanC
from utils import * #getListOfFolders

from skimage import measure
from scipy.misc import imread, imresize
from pandas import DataFrame

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

# thank you to: https://github.com/wcipriano/pretty-print-confusion-matrix
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix as cm
############################# Import Section #################################

def fetchClassName(dic, key):
	# print("Looking for " + str(key))
	for entry in dic.keys():
		# print("Looking in " + str(entry))
		if str(dic[entry]) == str(key):
			# input("Found!!")
			return entry;
		# print(str(dic[entry]) + " =/= " + str(key))
	return "NotFound..."

# Hyper Parameters
FRA = 4 # if Kitti: FRA = 9
PRE = 0 # represents predicting 1 frame
N_FRAME = FRA+PRE
T = FRA
numOfPixels = 240*320 # if Kitti: 128*160

gpu_id = 3
opticalflow_ckpt_file = '../preTrainedModel/UCFModel.pth' # if Kitti: 'KittiModel.pth'
classifier_ckpt_file = './2Stream-DYAN-ResNet86.pth' # './DyanC.pth'


stateDict = torch.load(opticalflow_ckpt_file)['state_dict']
Dtheta = stateDict['l1.theta']
Drr = stateDict['l1.rr']
baseDyan = OFModel(Drr, Dtheta, T, PRE, gpu_id)
baseDyan.cuda(gpu_id)
baseDyan.eval()

'''
loadedcheckpoint = torch.load(opticalflow_ckpt_file)
stateDict = loadedcheckpoint['state_dict']
        
# load parameters
Dtheta = stateDict['l1.theta'] 
Drr    = stateDict['l1.rr']
dyanEncoder = OFModel(Drr, Dtheta, FRA,PRE,gpu_id)
dyanEncoder.cuda(gpu_id)
'''

## Load the classifier network
classifier = unfrozen_DyanC(Drr, Dtheta, T, PRE, gpu_id)
classifier.load_state_dict(torch.load(classifier_ckpt_file)["state_dict"])
classifier.cuda(gpu_id)
classifier.eval();

ofSample = torch.FloatTensor(2, FRA, numOfPixels)

# set test list name:
validateFolderFile = 'validatelist05.txt'

# set test data directory
rootDirRGB = '/data/Abhishek/frames/' # RGB
rootDirOF = '/storage/truppr/UCF-FLOWS-FULL' # PGOF

# for UCF dataset:
validateFoldeList = getListOfFolders(validateFolderFile)[::3]

validateData = videoDatasetPreGenOFandRGB(folderList=validateFoldeList, rootDirPGOF=rootDirOF, rootDirRGB=rootDirRGB, N_FRAME=N_FRAME)

actionClasses = validateData.getACDic();
x = np.zeros((len(actionClasses.keys()),len(actionClasses.keys())))


''' print dictionary for ac
print()
for entry in actionClasses:
	actionClasses[entry] = actionClasses[entry] - 1;
	print (entry + " -- " + str(actionClasses[entry]))
'''

dataloader = DataLoader(validateData, batch_size=1, shuffle=True, num_workers=1)

for i_batch, sample in enumerate(dataloader):
	dataRGB = sample['frames'].squeeze(0).cuda(gpu_id)
	dataOF = sample['of'].squeeze(0).cuda(gpu_id)
	folderName = sample['ac']
	expectedOut = folderName.cuda(gpu_id);

	inputDataRGB = dataRGB.type(torch.FloatTensor).cuda(gpu_id).squeeze(0)
	inputDataOF = dataOF.type(torch.FloatTensor).cuda(gpu_id).squeeze(0)

	OF_3c = torch.zeros(inputDataOF.shape[0] + 1, inputDataOF.shape[1], inputDataOF.shape[2]).type(torch.FloatTensor).cuda(gpu_id)
	# print('data input size:', inputData.shape)
	# print('OF_3c size:', OF_3c.shape)
	# input()
	
	# 3 Channel OF representation
	OF_3c[2,:,:] = torch.sqrt(torch.mul(inputDataOF[0,:,:], inputDataOF[0,:,:]) + torch.mul(inputDataOF[1,:,:], inputDataOF[1,:,:]))
	OF_3c[0,:,:] = inputDataOF[0,:,:] / OF_3c[2,:,:];
	OF_3c[1,:,:] = inputDataOF[1,:,:] / OF_3c[2,:,:];
	
	OF_3c = Variable(OF_3c).type(torch.FloatTensor).cuda(gpu_id)

	# print("OF: ", OF_3c.shape)
	# print("RGB: ", inputDataRGB.shape)
	inputDataOF = baseDyan.forward2(OF_3c) # when DYAN is frozen...
	inputDataRGB = baseDyan.forward2(inputDataRGB)

	output = classifier.forward(inputDataRGB.squeeze(0), inputDataOF.squeeze(0))

	print("\nGround turth label -- " + fetchClassName(actionClasses, str(expectedOut.item())))
	print("Predicted label ----- " + fetchClassName(actionClasses, str(np.argmax(output.data.cpu().numpy()))))

	if fetchClassName(actionClasses, str(expectedOut.item())) == fetchClassName(actionClasses, str(np.argmax(output.data.cpu().numpy()))):
		print("Correct!!!")
	else:
		print("Failed!!!")
	
	x[expectedOut.item()][np.argmax(output.data.cpu().numpy())] = x[expectedOut.item()][np.argmax(output.data.cpu().numpy())] + 1;

	# print(output.shape)
	# print(np.argmax(output.data.cpu().numpy()))

print(x.shape)
for i in range(0, x.shape[0]):
	pass


trace = go.Heatmap(z=(x-np.min(x))/(np.max(x)-np.min(x)), x=sorted(actionClasses.keys()), y=sorted(actionClasses.keys()))
data=[trace]

# plotly.offline.plot({'data': data}, filename='20190416-unfrozen-32.html')

#get pandas dataframe
df_cm = DataFrame(x, index=range(0,x.shape[0]), columns=range(0,x.shape[0]))

#colormap: see this and choose your more dear
cmap = 'PuRd'
cm(df_cm, cmap=cmap);
