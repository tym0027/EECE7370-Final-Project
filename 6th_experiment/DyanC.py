############################# Import Section #################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torchvision
import torchvision.transforms.functional as tv
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from math import sqrt
import numpy as np

############################# UTILS Section #################################

gpu_id = 3;

def save_flow_to_img(flow, h, w, c):
	hsv = np.zeros((h, w, c), dtype=np.uint8)
	hsv[:, :, 0] = 255
	hsv[:, :, 2] = 255
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

	rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	# res_img_path = os.path.join('result', name)
	# cv2.imwrite(res_img_path, rgb)
	return rgb

############################# RESNet Section #################################

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None):
		super(BasicBlock, self).__init__()

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None):
		super(Bottleneck, self).__init__()

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups

		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=5, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
		super(ResNet, self).__init__()

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		self.inplanes = 64
		self.groups = groups
		self.base_width = width_per_group
		self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = norm_layer(self.inplanes)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		# self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		downsample = None
		
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):	
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		#x = self.fc(x)

		return x


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		state_dict = model_zoo.load_url(model_urls['resnet50'])
		#for param in model.parameters():
		#	param.requires_grad = False;
		'''	
		stdv = 1. / sqrt(model.fc.weight.size(1))
		model.fc.weight.data.uniform_(-stdv, stdv)
		if model.fc.bias is not None:
			model.fc.bias.data.uniform_(-stdv, stdv)
		'''
		s = {}
		for p in state_dict:
			if not p == "fc.weight" and not p == "fc.bias":
				s[p] = state_dict[p]
		# input(s)
		
		# s["fc.weight"] = 
		# s["fc.bias"] =
		
		# state_dict["fc.weight"] = 
		# state_dict["fc.bias"] = 
		# model.fc = nn.Linear(512, 101)
		model.load_state_dict(s)
	return model



# Create Dictionary
def creatRealDictionary(T,Drr,Dtheta,gpu_id):
        WVar = []
        Wones = torch.ones(1).cuda(gpu_id)
        Wones  = Variable(Wones,requires_grad=False)
        for i in range(0,T):
                W1 = torch.mul(torch.pow(Drr,i) , torch.cos(i * Dtheta))
                W2 = torch.mul ( torch.pow(-Drr,i) , torch.cos(i * Dtheta) )
                W3 = torch.mul ( torch.pow(Drr,i) , torch.sin(i *Dtheta) )
                W4 = torch.mul ( torch.pow(-Drr,i) , torch.sin(i*Dtheta) )
                W = torch.cat((Wones,W1,W2,W3,W4),0)

                WVar.append(W.view(1,-1))
        dic = torch.cat((WVar),0)
        G = torch.norm(dic,p=2,dim=0)
        idx = (G == 0).nonzero()
        nG = G.clone()
        nG[idx] = np.sqrt(T)
        G = nG

        dic = dic/G

        return dic


def fista(D, Y, lambd,maxIter,gpu_id):

	DtD = torch.matmul(torch.t(D),D)
	L = torch.norm(DtD,2)
	linv = 1/L
	DtY = torch.matmul(torch.t(D),Y)
	# print(DtY.shape)	
	x_old = Variable(torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id), requires_grad=True)
	t = 1
	y_old = x_old
	lambd = lambd*(linv.data.cpu().numpy())
	A = Variable(torch.eye(DtD.shape[1]).cuda(gpu_id),requires_grad=True) - torch.mul(DtD,linv)

	DtY = torch.mul(DtY,linv)

	Softshrink = nn.Softshrink(lambd )
	with torch.no_grad():
		for ii in range(maxIter):
		        Ay = torch.matmul(A,y_old)
		        del y_old
		        with torch.enable_grad():
		                x_new = Softshrink((Ay + DtY))
		        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
		        tt = (t-1)/t_new
		        y_old = torch.mul( x_new,(1 + tt))
		        y_old -= torch.mul(x_old , tt)
		        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-4:
		                x_old = x_new
		                break
		        t = t_new
		        x_old = x_new
		        del x_new
	return x_old


class Encoder(nn.Module):
        def __init__(self, Drr, Dtheta, T, gpu_id):
                super(Encoder, self).__init__()

                self.rr = nn.Parameter(Drr)
                self.theta = nn.Parameter(Dtheta)
                self.T = T
                self.gid = gpu_id

        def forward(self, x):
                dic = creatRealDictionary(self.T,self.rr,self.theta,self.gid)
                ## for UCF Dataset:
                sparsecode = fista(dic,x,0.1,100,self.gid)
                ## for Kitti Dataset: sparsecode = fista(dic,x,0.01,80,self.gid)
                return Variable(sparsecode)


class Decoder(nn.Module):
        def __init__(self,rr,theta, T, PRE, gpu_id ):
                super(Decoder,self).__init__()

                self.rr = rr
                self.theta = theta
                self.T = T
                self.PRE = PRE
                self.gid = gpu_id

        def forward(self,x):
                dic = creatRealDictionary(self.T+self.PRE,self.rr,self.theta,self.gid)
                result = torch.matmul(dic,x)
                return result


class OFModel(nn.Module):
        def __init__(self, Drr, Dtheta, T, PRE, gpu_id):
                super(OFModel, self).__init__()
                self.l1 = Encoder(Drr, Dtheta, T, gpu_id)
                self.l2 = Decoder(self.l1.rr,self.l1.theta, T, PRE, gpu_id)

        def forward(self,x):
                return self.l2(self.l1(x))

        def forward2(self,x):
                return self.l1(x)


class unfrozen_DyanC(nn.Module):
	# def __init__(self, T, PRE, trained_encoder, gpu_id):
	def __init__(self, Drr, Dtheta, T, PRE, gpu_id):
		super(unfrozen_DyanC, self).__init__()

		### Spatial Stream...
		self.ds_Conv1 = nn.Conv3d(1, 20,(161,1,1) )
		self.classify1 = resnet50(True)
		self.fc1 = nn.Linear(2048, 5)

		### Temporal Stream...
		self.ds_Conv2 = nn.Conv3d(1, 20,(161,1,1) )
		self.classify2 = resnet50(True)
		self.fc2 = nn.Linear(2048, 5)

		### Fully connected layer...
		# self.fc3 = nn.Linear(202, 101)

		### LogSoftmax...
		self.sm = nn.LogSoftmax(dim=1)

		i = 0;
		for param in self.parameters():
			# print(param)
			i = i + 1;
		print ("There are " + str(i) + " paramterss...")
	
	def forward(self, xRGB, xOF3):
		########## SPATIAL STREAM
		xRGB = xRGB.view(xRGB.shape[0],161,240,320).unsqueeze(1)
		# print("Xe re-shape: ", x.shape)	
		
		# Downsize latent space
		xRGB = self.ds_Conv1(xRGB); # Xeds
		# print("Xeds shape: ", x.shape)	

		xRGB = xRGB.squeeze(1)

		''' RGB Frames	'''
		x_ = xRGB.view(20, 3, 240, 320);
		x_ = x_.cpu()
		visualization = torch.zeros(3, 5 * 240, 4 * 320)
		for row in range(0, 5):	
			for col in range(0, 4):	
				visualization[:, row * 240 : (row * 240) + 240, col * 320 : (col * 320) + 320] = x_[4 * row + col,:,:, :] # RGB
		

		''' PGOF Frames '''
		'''
		x_ = x.view(20, 240, 320, 2);
		x_ = x_.cpu()
		visualization = torch.zeros(5 * 240, 4 * 320, 2)
		for row in range(0, 5):	
			for col in range(0, 4):	
				visualization[row * 240 : (row * 240) + 240, col * 320 : (col * 320) + 320, :] = x_[4 * row + col,:, :, :] # PGOF
		'''
		# print("Vis: ", visualization.shape)	
		
		''' For visualization... '''
		# a = tv.to_pil_image(save_flow_to_img(visualization.detach().cpu().numpy(), 240 * 5, 320 * 4,3)) # PGOF
		# a = tv.to_pil_image(visualization) # RGB
		# plt.imshow(a)
		# plt.show()
		''' END Visualization '''
		# input()

		xRGB = visualization.cuda(gpu_id);

		# Normalize for RESNet
		''' Ignore for now... As per Wen
		normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		x = normalize(x)
		'''

		# Resnet Clasify	
		xRGB = self.classify1(xRGB.unsqueeze(0))
		
		# print("ResNet Out: ", x.shape)

		# FC Layer
		xRGB = self.fc1(xRGB);
		# print("FC Layer Out: ", x.shape)

		########## TEMPEROL STREAM
		xOF3 = xOF3.view(xOF3.shape[0],161,240,320).unsqueeze(1)
		# print("Xe re-shape: ", x.shape)	
		
		# Downsize latent space
		xOF3 = self.ds_Conv2(xOF3); # Xeds
		# print("Xeds shape: ", x.shape)	

		xOF3 = xOF3.squeeze(1)

		''' RGB Frames	'''
		x_ = xOF3.view(20, 3, 240, 320);
		x_ = x_.cpu()
		visualization = torch.zeros(3, 5 * 240, 4 * 320)
		for row in range(0, 5):	
			for col in range(0, 4):	
				visualization[:, row * 240 : (row * 240) + 240, col * 320 : (col * 320) + 320] = x_[4 * row + col,:,:, :] # RGB
		

		''' PGOF Frames '''
		'''
		x_ = x.view(20, 240, 320, 2);
		x_ = x_.cpu()
		visualization = torch.zeros(5 * 240, 4 * 320, 2)
		for row in range(0, 5):	
			for col in range(0, 4):	
				visualization[row * 240 : (row * 240) + 240, col * 320 : (col * 320) + 320, :] = x_[4 * row + col,:, :, :] # PGOF
		'''
		# print("Vis: ", visualization.shape)	
		
		''' For visualization... '''
		# a = tv.to_pil_image(save_flow_to_img(visualization.detach().cpu().numpy(), 240 * 5, 320 * 4,3)) # PGOF
		# a = tv.to_pil_image(visualization) # RGB
		# plt.imshow(a)
		# plt.show()
		''' END Visualization '''
		# input()

		xOF3 = visualization.cuda(gpu_id);

		# Normalize for RESNet
		''' Ignore for now... As per Wen
		normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		x = normalize(x)
		'''

		# Resnet Clasify	
		xOF3 = self.classify2(xOF3.unsqueeze(0))
		
		# print("ResNet Out: ", x.shape)

		# FC Layer
		xOF3 = self.fc2(xOF3);
		# print("FC Layer Out: ", x.shape)

		########## FULLY CONNECTED
		x = torch.zeros(*xOF3.shape).cuda(gpu_id).squeeze(1)
		x = (xRGB + xOF3) / 2
		# x[0] = xRGB
		# x[1] =  
		print("X_final: ", x.shape)

		# x = x.view(1,-1)
		# print("X_final: ", x.shape)
		# x = self.fc3(x);

		# Return with LSoftMax
		return self.sm(x);
		
