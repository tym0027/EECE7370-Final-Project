## Imports related to PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import torchvision
import torchvision.transforms.functional as tv

## Imports related to tvnet
from model.losses.flow_loss import flow_loss
from model.net.tvnet import TVNet
from model.net.Conv2d_tensorflow import Conv2d
from model.net.spatial_transformer import spatial_transformer as st
from tvnet_utils import *

from math import sqrt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2


############################# UTILS Section #################################

GRAD_IS_ZERO = 1e-12

gpu_id = 2;

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


############################# Import Section #################################



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
	def __init__(self, block, layers, num_classes=101, zero_init_residual=False, groups=1, width_per_group=64, norm_layer=None):
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


class TVNet_Scale(nn.Module):
	def __init__(self, args, size):
		super(TVNet_Scale, self).__init__()

		self.tau = args.tau
		self.lbda = args.lbda
		self.theta = args.theta
		self.n_warps = args.n_warps
		self.zfactor = args.zfactor
		self.n_iters = args.n_iters
		self.data_size = size

		self.st = st()

		self.gradient_kernels = nn.ModuleList()
		self.divergence_kernels = nn.ModuleList()

		self.centered_gradient_kernels = self.get_centered_gradient_kernel().train(False)

		for n_warp in range(self.n_warps): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			gradient_warp = nn.ModuleList()
			divergence_warp = nn.ModuleList()
			
			for n_iter in range(self.n_iters):
				gradient_warp.append(get_module_list(self.get_gradient_kernel, 2))
				divergence_warp.append(get_module_list(self.get_divergence_kernel, 2))
			
			self.gradient_kernels.append(gradient_warp)
			self.divergence_kernels.append(divergence_warp)


	def get_gradient_kernel(self):
		gradient_block = nn.ModuleList()
		
		conv_x = Conv2d(1, 1, [1, 2], padding='SAME', bias=False, weight=[[[[-1, 1]]]])
		gradient_block.append(conv_x)
		conv_y = Conv2d(1, 1, [2, 1], padding='SAME', bias=False, weight=[[[[-1], [1]]]])
		gradient_block.append(conv_y)
		
		return gradient_block


	def get_divergence_kernel(self):
		divergence_block = nn.ModuleList() #[conv_x, conv_y]
		conv_x = Conv2d(1, 1, [1, 2], padding='SAME', bias=False, weight=[[[[-1, 1]]]])
		divergence_block.append(conv_x)

		conv_y = Conv2d(1, 1, [2, 1], padding='SAME', bias=False, weight=[[[[-1], [1]]]])
		divergence_block.append(conv_y)
		
		return divergence_block

	def get_centered_gradient_kernel(self):
		centered_gradient_block = nn.ModuleList()
		conv_x = Conv2d(1, 1, [1, 3], padding='SAME', bias=False, weight=[[[[-0.5, 0, 0.5]]]])
		centered_gradient_block.append(conv_x)
		conv_y = Conv2d(1, 1, [3, 1], padding='SAME', bias=False, weight=[[[[-0.5], [0], [0.5]]]])
		centered_gradient_block.append(conv_y)
		
		return centered_gradient_block

	def forward(self, x1, x2, u1, u2):
		assert x1.size() == u1.size(), "{} vs {}".format(x1.size(), u1.size())
		l_t = self.lbda * self.theta
		taut = self.tau / self.theta
		diff2_x, diff2_y = self.centered_gradient(x2)

		p11 = torch.zeros_like(x1).cuda(gpu_id)
		p12 = torch.zeros_like(x1).cuda(gpu_id)
		p21 = torch.zeros_like(x1).cuda(gpu_id)
		p22 = torch.zeros_like(x1).cuda(gpu_id)

		# p11 = p12 = p21 = p22 = tf.zeros_like(x1) in original tensorflow code, 
		# it seems that each element of p11 to p22 shares a same memory address and I'm not sure if it would make some mistakes or not.
	
		for n_warp in range(self.n_warps):
			u1_flat = u1.view(x2.size(0), 1, x2.size(2)*x2.size(3))
			u2_flat = u2.view(x2.size(0), 1, x2.size(2)*x2.size(3))

			x2_warp = self.warp_image(x2, u1_flat, u2_flat)
			x2_warp = x2_warp.view(x2.size())

			diff2_x_warp = self.warp_image(diff2_x, u1_flat, u2_flat)
			diff2_x_warp = diff2_x_warp.view(diff2_x.size())
			# print(diff2_x_warp.size())

			diff2_y_warp = self.warp_image(diff2_y, u1_flat, u2_flat)
			diff2_y_warp = diff2_y_warp.view(diff2_y.size())

			diff2_x_sq = diff2_x_warp ** 2
			diff2_y_sq = diff2_y_warp ** 2

			grad = diff2_x_sq + diff2_y_sq + GRAD_IS_ZERO
			rho_c = x2_warp.cuda(gpu_id).type(torch.cuda.FloatTensor) - diff2_x_warp.cuda(gpu_id).type(torch.cuda.FloatTensor) * u1 - diff2_y_warp.cuda(gpu_id).type(torch.cuda.FloatTensor) * u2 - x1.cuda(gpu_id).type(torch.cuda.FloatTensor)

			for n_iter in range(self.n_iters):
				rho = rho_c + diff2_x_warp * u1 + diff2_y_warp * u2 + GRAD_IS_ZERO
				masks1 = rho < -l_t * grad
				d1_1 = torch_where(masks1, l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
				d2_1 = torch_where(masks1, l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))
				
				masks2 = rho > l_t * grad
				d1_2 = torch_where(masks2, -l_t * diff2_x_warp, torch.zeros_like(diff2_x_warp))
				d2_2 = torch_where(masks2, -l_t * diff2_y_warp, torch.zeros_like(diff2_y_warp))

				masks3 = (rho >= -l_t * grad) & (rho <= l_t * grad) & (grad > GRAD_IS_ZERO)
				d1_3 = torch_where(masks3, -rho / grad * diff2_x_warp, torch.zeros_like(diff2_x_warp))
				d2_3 = torch_where(masks3, -rho / grad * diff2_y_warp, torch.zeros_like(diff2_y_warp))

				v1 = d1_1 + d1_2 + d1_3 + u1
				v2 = d2_1 + d2_2 + d2_3 + u2

				u1 = v1 + self.theta * self.forward_divergence(p11, p12, n_warp, n_iter, 0)
				u2 = v2 + self.theta * self.forward_divergence(p21, p22, n_warp, n_iter, 1)

				u1x, u1y = self.forward_gradient(u1, n_warp, n_iter, 0)
				u2x, u2y = self.forward_gradient(u2, n_warp, n_iter, 1)

				p11 = (p11 + taut * u1x) / (
					1.0 + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + GRAD_IS_ZERO))
				p12 = (p12 + taut * u1y) / (
					1.0 + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + GRAD_IS_ZERO))
				p21 = (p21 + taut * u2x) / (
					1.0 + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + GRAD_IS_ZERO))
				p22 = (p22 + taut * u2y) / (
					1.0 + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + GRAD_IS_ZERO))
		return u1, u2, rho

	def centered_gradient(self, x):
		assert len(x.shape) == 4
		diff_x = self.centered_gradient_kernels[0](x)
		diff_y = self.centered_gradient_kernels[1](x)

		# refine the boundary
		first_col = 0.5 * (x[..., 1:2] - x[..., 0:1])
		last_col = 0.5 * (x[..., -1:] - x[..., -2:-1])
		diff_x_valid = diff_x[..., 1:-1]
		diff_x = torch.cat([first_col.type(torch.FloatTensor), diff_x_valid.type(torch.FloatTensor), last_col.type(torch.FloatTensor)], dim=-1)
		first_row = 0.5 * (x[:, :, 1: 2, :] - x[:, :, 0: 1, :])
		last_row = 0.5 * (x[:, :, -1:, :] - x[:, :, -2:-1, :])
		diff_y_valid = diff_y[:, :, 1:-1, :]
		diff_y = torch.cat([first_row.type(torch.FloatTensor), diff_y_valid.type(torch.FloatTensor), last_row.type(torch.FloatTensor)], dim=-2)

		return diff_x, diff_y

	def warp_image(self, x, u, v):
		assert len(x.size()) == 4
		assert len(u.size()) == 3
		assert len(v.size()) == 3

		u = u / x.size(3) * 2
		v = v / x.size(2) * 2
		theta = torch.cat((u, v), dim=1).cuda(gpu_id).float()
		trans_image = self.st(x, theta, x.size()[2:])

		return trans_image

	def forward_divergence(self, x, y, n_warp, n_iter, n_kernel):
		assert len(x.size()) == 4 #[bs, c, h, w]
		assert x.size(1) == 1 # grey scale image

		x_valid = x[:, :, :, :-1]
		first_col = Variable(torch.zeros(x.size(0), x.size(1), x.size(2), 1).float().cuda(gpu_id))
		x_pad = torch.cat((first_col, x_valid), dim=3)

		y_valid = y[:, :, :-1, :]
		first_row = Variable(torch.zeros(y.size(0), y.size(1), 1, y.size(3)).float().cuda(gpu_id))
		y_pad = torch.cat((first_row, y_valid), dim=2)

		diff_x = self.divergence_kernels[n_warp][n_iter][n_kernel][0](x_pad)
		diff_y = self.divergence_kernels[n_warp][n_iter][n_kernel][1](y_pad)

		div = diff_x + diff_y
		
		return div


	def forward_gradient(self, x, n_warp, n_iter, n_kernel):
		assert len(x.size()) == 4
		assert x.size(1) == 1 # grey scale image

		diff_x = self.gradient_kernels[n_warp][n_iter][n_kernel][0](x)
		diff_y = self.gradient_kernels[n_warp][n_iter][n_kernel][1](x)

		diff_x_valid = diff_x[:, :, :, :-1]
		last_col = Variable(torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), diff_x_valid.size(2), 1).float().cuda(gpu_id))
		diff_x = torch.cat((diff_x_valid, last_col), dim=3)

		diff_y_valid = diff_y[:, :, :-1, :]
		last_row = Variable(torch.zeros(diff_x_valid.size(0), diff_x_valid.size(1), 1, diff_y_valid.size(3)).float().cuda(gpu_id))
		diff_y = torch.cat((diff_y_valid, last_row), dim=2)

		return diff_x, diff_y


class unfrozen_DyanC(nn.Module):
	# def __init__(self, T, PRE, trained_encoder, gpu_id):
	def __init__(self, Drr, Dtheta, T, PRE, tvnet_args, gpu_id):
		super(unfrozen_DyanC, self).__init__()
		self.args = tvnet_args
		self.data_size = tvnet_args.data_size
		self.zfactor = tvnet_args.zfactor
		self.max_nscale = tvnet_args.max_nscale

		_, _, self.height, self.width = self.data_size
		n_scales = 1 + np.log(np.sqrt(self.height ** 2 + self.width ** 2) / 4.0) / np.log(1 / self.zfactor)
		self.n_scales = min(n_scales, self.max_nscale)

		self.st = st()
		
		self.tvnet_kernels = nn.ModuleList()

		for ss in range(self.n_scales):
			self.tvnet_kernels.append(TVNet_Scale(tvnet_args, self.data_size))

		self.gray_kernel = self.get_gray_conv().train(False)
		self.gaussian_kernel = self.get_gaussian_conv().train(False)

		u_size = self.zoom_size(self.data_size[2], self.data_size[3], self.zfactor ** (self.max_nscale - 1))
		self.u1_init = nn.Parameter(torch.zeros(1, 1, *u_size).float())
		self.u2_init = nn.Parameter(torch.zeros(1, 1, *u_size).float())

		# load parameters for DYAN
		# self.Drr = nn.Parameter(Drr)
		# self.Dtheta = nn.Parameter(Dtheta)
		self.l1 = Encoder(Drr, Dtheta, T, gpu_id)
		
		# Downsize image...
		self.ds_Conv = nn.Conv3d(1, 20,(161,1,1) )
			

		self.classify = resnet50(True)

		self.fc = nn.Linear(2048, 101)

		self.sm = nn.LogSoftmax(dim=1)

		i = 0;
		for param in self.parameters():
			# print(param)
			i = i + 1;
		print ("There are " + str(i) + " paramterss...")
		
		# self.classifier = classifier
	
	def forward(self, data):
		with torch.no_grad():
			x1 = torch.zeros(1, data.shape[1], data.shape[2], data.shape[3]);
			x2 = torch.zeros(1, data.shape[1], data.shape[2], data.shape[3]);

			for i in range(0, data.shape[1]):
				x1[0,:,:,:] = data[i,:,:,:]
				x2[0,:,:,:] = data[i + 1,:,:,:]

			OF = torch.FloatTensor(2, x1.shape[1] + 1, x1.shape[2] * x1.shape[3]).cuda(gpu_id)

			if x1.size(1) == 3:
				x1 = self.gray_scale_image(x1)
				x2 = self.gray_scale_image(x2)

			smooth_x1 = torch.zeros(x1.shape)
			smooth_x2 = torch.zeros(x2.shape)

			for i in range(0, x1.shape[0]):
				norm_imgs = self.normalize_images(x1[i,:,:,:].unsqueeze(0), x2[i,:,:,:].unsqueeze(0))
				smooth_x1[i,:,:,:] = self.gaussian_smooth(norm_imgs[0])
				smooth_x2[i,:,:,:] = self.gaussian_smooth(norm_imgs[1])

			for ss in range(self.n_scales-1, -1, -1):
				# print("\titerating over scales...")
				down_sample_factor = self.zfactor ** ss
				down_height, down_width = self.zoom_size(self.height, self.width, down_sample_factor)

				# print("idk shape")
				# input(.shape)

				down_x1 = torch.zeros(x1.shape[0], x1.shape[1], down_height, down_width)
				down_x2 = torch.zeros(x2.shape[0], x2.shape[1], down_height, down_width)

				# To remove forloop unindent from under this forloop
				for i in range(0, x1.shape[0]):
					u1, u2 = self.u1_init.repeat(1, 1, 1, 1), self.u2_init.repeat(1, 1, 1, 1)

					u1 = u1.cuda(gpu_id).type(torch.cuda.FloatTensor)
					u2 = u2.cuda(gpu_id).type(torch.cuda.FloatTensor)

					down_x1[i,:,:,:] = self.zoom_image(smooth_x1[i,:,:,:].unsqueeze(0), down_height, down_width)
					down_x2[i,:,:,:] = self.zoom_image(smooth_x2[i,:,:,:].unsqueeze(0), down_height, down_width)

					# down_x1 = self.zoom_image(smooth_x1, down_height, down_width)
					# down_x2 = self.zoom_image(smooth_x2, down_height, down_width)

					# print("down_x1 shape")
					# input(down_x1.shape)

					u1, u2, rho = self.tvnet_kernels[ss](down_x1[i,:,:,:].unsqueeze(0), down_x2[i,:,:,:].unsqueeze(0), u1, u2)
					# print("u2 shape")
					# input(u2.shape)
			
					if ss == 0:
						# you don't want to return here, you want to break and set OF
						# return u1, u2, rho
						OF[0] = u1.view(x1.shape[2] * x1.shape[3])
						OF[1] = u2.view(x1.shape[2] * x1.shape[3])
				
						# print("OF:")
						# print(OF.shape)
						# print(OF)
						# input();
				
						break

					up_sample_factor = self.zfactor ** (ss - 1)
					up_height, up_width = self.zoom_size(self.height, self.width, up_sample_factor)
					u1 = self.zoom_image(u1, up_height, up_width) / self.zfactor
					u2 = self.zoom_image(u2, up_height, up_width) / self.zfactor
		
		# 3 channel optical flow
		OF_3c = torch.zeros(OF.shape[0] + 1, OF.shape[1], OF.shape[2]).type(torch.FloatTensor).cuda(gpu_id)
		OF_3c[2,:,:] = torch.sqrt(torch.mul(OF[0,:,:], OF[0,:,:]) + torch.mul(OF[1,:,:], OF[1,:,:]))
		OF_3c[0,:,:] = OF[0,:,:] / OF_3c[2,:,:];
		OF_3c[1,:,:] = OF[1,:,:] / OF_3c[2,:,:];
		OF_3c = Variable(OF_3c).type(torch.FloatTensor).cuda(gpu_id)

		# print("OF pre-view shape: ", OF.shape)
		# OF = OF.view(4, 240, 320, 2)
		# print("OF post-view shape: ", OF[2,:,:,:].shape)
		# im = save_flow_to_img(OF[2,:,:,:].squeeze(0).detach().cpu().numpy(), 240, 320, 3)
		# a = tv.to_pil_image(im)
		
		# plt.imshow(im)
		# plt.show()
		# input()
		
		''' RGB Frames	'''
		# print(OF_3c.shape)
		# s = OF_3c[:,1,:].cpu().view(240, 320, 3)
		# print("S: ", s.shape)
		# a = tv.to_pil_image(s) # RGB
		# print("A: ", a.size)
		# plt.imshow(s)
		# plt.show()
		''' END Visualization '''
		# input()
		# OF_3c[OF_3c != OF_3c] = 0;
		'''
		print("OF pre-view shape: ", OF.shape)
		OF = OF.view(4, 240, 320, 2)
		# input(OF[1,:,:,:].squeeze(0).shape)
		print("OF post-view shape: ", OF[2,:,:,:].shape)
		print("OF post-view w/ squeeze shape: ", OF[2,:,:,:].squeeze(0).shape)	
		im = save_flow_to_img(OF[2,:,:,:].squeeze(0).detach().cpu().numpy(), 240, 320, 3)
		print("Im shape: ", im.shape)
		# a = tv.to_pil_image(im)
		# plt.imshow(a)
		# plt.show()
		# input()
		'''

		# Encoder:
		# print(OF_3c.shape)
		x = self.l1(OF_3c)
		x = x.unsqueeze(dim=0)
		# print(x.shape)
		
		x = x.view(x.shape[1],161,240,320).unsqueeze(1)
		# print("Xe re-shape: ", x.shape)	
		# Downsize latent space
		x = self.ds_Conv(x); # Xeds
		# print("Xeds shape: ", x.shape)	

		x = x.squeeze(1)
				
		''' RGB Frames	'''
		x_ = x.view(20, 3, 240, 320);
		x_ = x_.cpu()
		# print(x_)				
		visualization = torch.zeros(3, 5 * 240, 4 * 320)
		for row in range(0, 5):	
			for col in range(0, 4):	
				visualization[:, row * 240 : (row * 240) + 240, col * 320 : (col * 320) + 320] = x_[4 * row + col,:,:, :] # RGB

		# print("Vis: ", visualization.shape)	
		''' For visualization... '''
		# a = tv.to_pil_image(save_flow_to_img(visualization.detach().cpu().numpy(), 240 * 5, 320 * 4,3)) # PGOF
		# a = tv.to_pil_image(visualization) # RGB
		# plt.imshow(a)
		# plt.show()
		''' END Visualization '''
		# input()

		x = visualization.cuda(gpu_id);
		
		# print(x.shape)

		# Normalize for RESNet
		''' Ignore for now... As per Wen
		normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		x = normalize(x)
		'''

		# print(x.shape)

		# Resnet Clasify	
		x = self.classify(x.unsqueeze(0))
		
		# print("ResNet Out: ", x.shape)

		# FC Layer
		x = self.fc(x);
		# print("FC Layer Out: ", x.shape)

		# Return with LSoftMax
		return self.sm(x);


	def get_gray_conv(self):
		gray_conv = Conv2d(3, 1, [1, 1], bias=False, padding='SAME',
			weight=[[[[0.114]], [[0.587]], [[0.299]]]])

		return gray_conv


	def get_gaussian_conv(self):
		gaussian_conv = Conv2d(1, 1, [5, 5], bias=False, padding='SAME',
			weight=[[[[0.000874, 0.006976, 0.01386, 0.006976, 0.000874],
				[0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
				[0.01386, 0.110656, 0.219833, 0.110656, 0.01386],
				[0.006976, 0.0557, 0.110656, 0.0557, 0.006976],
				[0.000874, 0.006976, 0.01386, 0.006976, 0.000874]]]])
		return gaussian_conv


	def gray_scale_image(self, x):
		assert len(x.size()) == 4
		assert x.size(1) == 3, 'number of channels must be 3 (i.e. RGB)'

		gray_x = self.gray_kernel(x)
		return gray_x


	def gaussian_smooth(self, x):
		assert len(x.size()) == 4
		smooth_x = self.gaussian_kernel(x)

		return smooth_x


	def normalize_images(self, x1, x2):
		# print(x1)
		# print(x1.shape)
		# input("HELLO!")

		
		min_x1 = x1.min(3)[0].min(2)[0].min(1)[0]
		max_x1 = x1.max(3)[0].max(2)[0].max(1)[0]

		min_x2 = x2.min(3)[0].min(2)[0].min(1)[0]
		max_x2 = x2.max(3)[0].max(2)[0].max(1)[0]
	
		min_val = torch.min(min_x1, min_x2)
		max_val = torch.max(max_x1, max_x2)

		den = max_val - min_val
		
		expand_dims = [-1 if i == 0 else 1 for i in range(len(x1.shape))]
		min_val_ex = min_val.view(*expand_dims)
		den_ex = den.view(*expand_dims)

		x1_norm = torch_where(den > 0, 255. * (x1 - min_val_ex) / den_ex, x1)
		x2_norm = torch_where(den > 0, 255. * (x2 - min_val_ex) / den_ex, x2)
		return x1_norm, x2_norm
		

		''' TODO: WORK ON ALL INDECES AT ONCE
		min_x1 = x1[:,:,:,:].min(3)[0].min(2)[0].min(1)[0]
		max_x1 = x1[:,:,:,:].max(3)[0].max(2)[0].max(1)[0]

		min_x2 = x2[:,:,:,:].min(3)[0].min(2)[0].min(1)[0]
		max_x2 = x2[:,:,:,:].max(3)[0].max(2)[0].max(1)[0]
        
		min_val = torch.min(min_x1, min_x2)
		max_val = torch.max(max_x1, max_x2)

		den = max_val - min_val
               
		# input("we got this far iddn't we?")
 
		expand_dims = [-1 if i == 0 else 1 for i in range(len(x1.shape))]
		min_val_ex = min_val.view(*expand_dims)
		den_ex = den.view(*expand_dims)

		x1_norm = torch.zeros(x1.shape)
		x2_norm = torch.zeros(x2.shape)
	
		input(x1)
	
		for i in range(0, x1.shape[0]):
			x1_norm[i,:,:,:] = torch_where(den[i] > 0, 255. * (x1[i,:,:,:] - min_val_ex[i]) / den_ex[i], x1[i,:,:,:])
			x2_norm[i,:,:,:] = torch_where(den[i] > 0, 255. * (x2[i,:,:,:] - min_val_ex[i]) / den_ex[i], x2[i,:,:,:])
		
		input(x1_norm)

		return x1_norm, x2_norm
		'''

	def zoom_size(self, height, width, factor):
		new_height = int(float(height) * factor + 0.5)
		new_width = int(float(width) * factor + 0.5)

		return new_height, new_width

	
	def zoom_image(self, x, new_height, new_width):
		assert len(x.shape) == 4
		theta = Variable(torch.zeros(x.size(0), 2, new_height * new_width).cuda(gpu_id).float())
		zoomed_x = self.st(x, theta, (new_height, new_width))
		return zoomed_x.view(x.size(0), x.size(1), new_height, new_width)
