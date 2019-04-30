############################# Import Section #################################
## Generic imports
import os
import time
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import re

## Imports related to PyTorch
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
############################# Import Section #################################

## Dataloader for PyTorch.
class videoDatasetPreGenOFandRGB(Dataset):
	"""Dataset Class for Loading Video"""

	def __init__(self, folderList, rootDirPGOF, rootDirRGB, N_FRAME):

		"""
		Args:
			N_FRAME (int) : Number of frames to be loaded
			rootDir (string): Directory with all the Frames/Videoes.
			Image Size = 240,320
			2 channels : U and V
		"""

		self.listOfFolders = folderList
		self.rootDirPGOF = rootDirPGOF
		self.rootDirRGB = rootDirRGB
		self.nfra = N_FRAME
		self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
		self.action_classes = {}
		self.action_classes = self.actionClassDictionary();
		print("Loaded " + str(len(self.action_classes)) + " action classes...")

	def __len__(self):
		return len(self.listOfFolders)

	def getACDic(self):
		return self.action_classes;

	def actionClassDictionary(self):
		num_classes = 0;
		for folderName in sorted(self.listOfFolders):
			result = re.search('v_(.*)_g', folderName)
			n = result.group(1)
			if n in self.action_classes.keys():
				continue;
			else:
				# print(str(n) + " -- " + str(num_classes))
				self.action_classes[n] = num_classes;
				num_classes = num_classes + 1;

		return self.action_classes

	def readData(self, folderName):
		path = os.path.join(self.rootDirPGOF,folderName)
		OF = torch.FloatTensor(2,self.nfra,self.numpixels)
		for framenum in range(self.nfra):
			flow = np.load(os.path.join(path,str(framenum)+'.npy'))
			flow = np.transpose(flow,(2,0,1))
			OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
		
		path = os.path.join(self.rootDirRGB,folderName)
		frames = torch.FloatTensor(3,self.nfra,self.numpixels)

		### NEED N + 1 frames when starting with raw frames
		frames = torch.zeros(3, self.nfra, 240*320);
		# frames = torch.zeros(self.nfra, 3, 240, 320)
		for framenum in range(1, self.nfra):
			
			# print("reading from frame " + str(framenum) + "...")
			img_path1 = os.path.join(path, "image-" + str('%04d' % framenum) + '.jpeg')		
			image1 = Image.open(img_path1)
			image1 = ToTensor()(image1)
			image1 = image1.float()		
			# print(image1.shape)
			image1 = image1.view(-1, 240*320)
			# print(image1.shape)			
			
			frames[:,framenum,:] = image1

		return OF, frames

	def __getitem__(self, idx):
		folderName = self.listOfFolders[idx]

		result = re.search('v_(.*)_g', folderName)
		n = result.group(1)
		if n in self.action_classes.keys():
			ac = self.action_classes[n]
		else:
			input("Found new action class???")
			self.action_classes[n] = self.num_classes
			self.num_classes = self.num_classes + 1;
			ac = self.action_classes[n]

		OF, Frame = self.readData(folderName)
		sample = { 'frames': Frame ,  'of':OF, 'ac' : ac }

		return sample

## Dataloader for PyTorch.
class videoDatasetRawFrames(Dataset):
	"""Dataset Class for Loading Video"""

	def __init__(self, folderList, rootDir, N_FRAME):

		"""
		Args:
			N_FRAME (int) : Number of frames to be loaded
			rootDir (string): Directory with all the Frames/Videoes.
			Image Size = 240,320
			2 channels : U and V
		"""

		self.listOfFolders = folderList
		self.rootDir = rootDir
		self.nfra = N_FRAME
		self.numpixels = 240*320 # If Kitti dataset, self.numpixels = 128*160
		self.action_classes = {}
		self.action_classes = self.actionClassDictionary();
		print("Loaded " + str(len(self.action_classes)) + " action classes...")

	def __len__(self):
		return len(self.listOfFolders)

	def getACDic(self):
		return self.action_classes;

	def actionClassDictionary(self):
		num_classes = 0;
		for folderName in sorted(self.listOfFolders):
			result = re.search('v_(.*)_g', folderName)
			n = result.group(1)
			if n in self.action_classes.keys():
				continue;
			else:
				# print(str(n) + " -- " + str(num_classes))
				self.action_classes[n] = num_classes;
				num_classes = num_classes + 1;

		return self.action_classes

	def readRGBData(self, folderName):
		# path = os.path.join(self.rootDir,folderName)
		# img = torch.FloatTensor(3, self.nfra,self.numpixels)
		path = os.path.join(self.rootDir,folderName)
		frames = torch.FloatTensor(3,self.nfra,self.numpixels)

		offset = len([name for name in sorted(os.listdir(path)) if ".jpeg" in name]);
		offset = random.randint(1, offset - self.nfra - 1)

		# print("reading " + str(self.nfra)  + " data")

		### NEED N + 1 frames when starting with raw frames
		frames = torch.zeros(3, self.nfra, 240*320);
		# frames = torch.zeros(self.nfra, 3, 240, 320)
		for framenum in range(offset, self.nfra + offset):
			
			# print("reading from frame " + str(framenum) + "...")
			img_path1 = os.path.join(path, "image-" + str('%04d' % framenum) + '.jpeg')		
			image1 = Image.open(img_path1)
			image1 = ToTensor()(image1)
			image1 = image1.float()		
			# print(image1.shape)
			image1 = image1.view(-1, 240*320)
			# print(image1.shape)			
			
			frames[:,framenum - offset,:] = image1
		# print(frames.shape)	
		return frames

	def readOFData(self, folderName):
		path = os.path.join(self.rootDir,folderName)
		OF = torch.FloatTensor(2,self.nfra,self.numpixels)
		for framenum in range(self.nfra):
			flow = np.load(os.path.join(path,str(framenum)+'.npy'))
			flow = np.transpose(flow,(2,0,1))
			OF[:,framenum] = torch.from_numpy(flow.reshape(2,self.numpixels)).type(torch.FloatTensor)
		
		return OF

	def __getitem__(self, idx):
		folderName = self.listOfFolders[idx]

		result = re.search('v_(.*)_g', folderName)
		n = result.group(1)
		if n in self.action_classes.keys():
			ac = self.action_classes[n]
		else:
			input("Found new action class???")
			self.action_classes[n] = self.num_classes
			self.num_classes = self.num_classes + 1;
			ac = self.action_classes[n]

		Frame = self.readRGBData(folderName)
		sample = { 'frames': Frame , 'ac' : ac }

		return sample


## Design poles

def gridRing(N):
	epsilon_low = 0.25
	epsilon_high = 0.15
	rmin = (1-epsilon_low)
	rmax = (1+epsilon_high)
	thetaMin = 0.001
	thetaMax = np.pi/2 - 0.001
	delta = 0.001
	Npole = int(N/4)
	Pool = generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax)
	M = len(Pool)
	idx = random.sample(range(0, M), Npole)
	P = Pool[idx]
	Pall = np.concatenate((P,-P, np.conjugate(P),np.conjugate(-P)),axis = 0)

	return P,Pall

## Generate the grid on poles
def generateGridPoles(delta,rmin,rmax,thetaMin,thetaMax):
	rmin2 = pow(rmin,2)
	rmax2 = pow(rmax,2)
	xv = np.arange(-rmax,rmax,delta)
	x,y = np.meshgrid(xv,xv,sparse = False)
	mask = np.logical_and( np.logical_and(x**2 + y**2 >= rmin2 , x**2 + y **2 <= rmax2),
						   np.logical_and(np.angle(x+1j*y)>=thetaMin, np.angle(x+1j*y)<=thetaMax ))
	px = x[mask]
	py = y[mask]
	P = px + 1j*py
	
	return P


# Create Gamma for Fista
def getWeights(Pall,N):
	g2 = pow(abs(Pall),2)
	g2N = np.power(g2,N)

	GNum = 1-g2
	GDen = 1-g2N
	idx = np.where(GNum == 0)[0]

	GNum[idx] = N
	GDen[idx] = pow(N,2)
	G = np.sqrt(GNum/GDen)
	return np.concatenate((np.array([1]),G))

## Functions for printing time
def asMinutes(s):
	m = math.floor(s / 60)
	s -= m * 60
	return '%dm %ds' % (m, s)


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

## Function to save the checkpoint
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


def getListOfFolders(File):
	data = pd.read_csv(File, sep=" ", header=None)[0]
	data = data.str.split('/',expand=True)[1]
	data = data.str.rstrip(".avi").values.tolist()

	return data
