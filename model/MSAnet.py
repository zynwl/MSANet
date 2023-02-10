# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 19:
# @Author  : zyn
# @Email : zyn962464@gmail
# @FileName: MSAnet.py
import pywt
import math
import os
import numpy as np
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.laplace
from torch.autograd import Function


def make_model(args, parent=False):
    return Mainnet(args)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kernel = io.loadmat('./init_kernel.mat')['C5']
kernel = torch.FloatTensor(kernel)
kernel4 = torch.FloatTensor(kernel[:, :16, 0:5, 0:5])
kernel5 = torch.FloatTensor(torch.ones(3, 9, 1, 1) / 1000)

w_x = (torch.FloatTensor(
    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])) / 1000
w_x1 = (torch.FloatTensor([1.0])) / 1000
w_x_conv = w_x.unsqueeze(dim=0).unsqueeze(dim=0)
w_x1_conv = w_x1.unsqueeze(dim=0).unsqueeze(dim=0)


class Mainnet(nn.Module):
    def __init__(self, args):
        super(Mainnet, self).__init__()
        self.S = args.stage  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        self.num_M = 48
        self.num_Z = 48
        self.num_Z1 = 48
        self.wavename = 'haar'
        # Stepsize
        # initialization
        self.etaM1 = torch.Tensor([1])
        self.etaM2 = torch.Tensor([1])
        self.etaM3 = torch.Tensor([1])
        self.etaM4 = torch.Tensor([1])
        # initialization
        self.etaM5 = torch.Tensor([1])
        self.etaM6 = torch.Tensor([1])
        self.etaM7 = torch.Tensor([1])
        self.etaM8 = torch.Tensor([0.1])
        # initialization
        self.etaM9 = torch.Tensor([1])
        self.etaM10 = torch.Tensor([1])
        self.etaM11 = torch.Tensor([1])
        self.etaM12 = torch.Tensor([1])
        self.etaM13 = torch.Tensor([1])
        self.etaX = torch.Tensor([1])
        self.etaX1 = torch.Tensor([1])
        self.etaX2 = torch.Tensor([1])
        # initialization
        self.etaX3 = torch.Tensor([1])
        self.etaU = torch.Tensor([1])
        self.etau = torch.Tensor([10])
        self.etae = torch.Tensor([10])
        self.etae1 = torch.Tensor([10])
        self.etae2 = torch.Tensor([10])
        # usd in initialization process
        self.eta1 = nn.Parameter(self.etaM1, requires_grad=True)
        # usd in initialization process
        self.eta2 = nn.Parameter(self.etaX, requires_grad=True)
        self.eta3 = nn.Parameter(self.etaU, requires_grad=True)
        self.eta4 = nn.Parameter(self.etaM2, requires_grad=True)
        # usd in initialization process
        self.eta5 = nn.Parameter(self.etaM3, requires_grad=True)
        self.eta6 = nn.Parameter(self.etaM4, requires_grad=True)
        # usd in initialization process
        self.eta7 = nn.Parameter(self.etaM5, requires_grad=True)
        # usd in initialization process
        self.eta8 = nn.Parameter(self.etaM6, requires_grad=True)
        self.eta9 = nn.Parameter(self.etaM7, requires_grad=True)
        self.eta10 = nn.Parameter(self.etaM8, requires_grad=True)
        # usd in initialization process
        self.eta21 = nn.Parameter(self.etaM9, requires_grad=True)
        self.eta22 = nn.Parameter(self.etaM10, requires_grad=True)
        self.eta23 = nn.Parameter(self.etaX1, requires_grad=True)
        # usd in initialization process
        self.eta24 = nn.Parameter(self.etaX2, requires_grad=True)
        self.eta25 = nn.Parameter(self.etaX3, requires_grad=True)
        self.eta35 = nn.Parameter(self.etau, requires_grad=True)
        self.eta45 = nn.Parameter(self.etae, requires_grad=True)
        self.eta55 = nn.Parameter(self.etae1, requires_grad=True)
        self.eta65 = nn.Parameter(self.etae2, requires_grad=True)
        # usd in iterative process
        self.eta11 = self.make_eta1(self.iter, self.etaM1)
        # usd in iterative process
        self.eta12 = self.make_eta(self.iter, self.etaX)
        self.eta13 = self.make_eta(self.iter, self.etaU)
        # usd in iterative process
        self.eta14 = self.make_eta(self.iter, self.etaM2)
        self.eta15 = self.make_eta(self.iter, self.etaM3)
        self.eta16 = self.make_eta(self.iter, self.etaM4)
        # usd in iterative process
        self.eta17 = self.make_eta(self.iter, self.etaM5)
        # usd in iterative process
        self.eta18 = self.make_eta(self.iter, self.etaM6)
        self.eta19 = self.make_eta(self.iter, self.etaM7)
        # usd in iterative process
        self.eta110 = self.make_eta(self.iter, self.etaM8)
        self.eta111 = self.make_eta(self.iter, self.etaM9)
        self.eta112 = self.make_eta(self.iter, self.etaM10)
        # usd in iterative process
        self.eta113 = self.make_eta(self.iter, self.etaX1)
        self.eta114 = self.make_eta(self.iter, self.etaX2)
        self.eta115 = self.make_eta(self.iter, self.etaX3)
        self.eta112 = self.make_eta(self.iter, self.etae)
        self.eta117 = self.make_eta(self.iter, self.etau)
        self.eta118 = self.make_eta(self.iter, self.etaM12)
        self.eta119 = self.make_eta(self.iter, self.etaM13)
        self.eta120 = self.make_eta(self.iter, self.etae1)
        self.eta121 = self.make_eta(self.iter, self.etae2)
        # Rain kernel
        w_z_f0 = w_x1_conv.expand(3, 9, -1, -1)
        # used in initialization process
        self.weight0 = nn.Parameter(data=kernel, requires_grad=True)
        self.weight1 = nn.Parameter(data=w_z_f0, requires_grad=True)
        #self.weight2 = nn.Parameter(data=kernel2, requires_grad = True)
        #self.weight3 = nn.Parameter(data=kernel3, requires_grad = True)
        self.weight4 = nn.Parameter(data=kernel4, requires_grad=True)
        self.weight5 = nn.Parameter(data=kernel5, requires_grad=True)

        # rain kernel is inter-stage sharing. The true net parameter number is
        # (#self.conv /self.iter)
        self.conv = self.make_weight(self.iter, kernel)
        self.conv1 = self.make_weight(self.iter, w_z_f0)
        #self.conv2 = self.make_weight(self.iter, kernel2)
        #self.conv3 = self.make_weight(self.iter, kernel3)
        self.conv4 = self.make_weight(self.iter, kernel4)
        self.conv5 = self.make_weight(self.iter, kernel5)

        # filter for initializing B and Z
        self.w_z_f0 = w_x_conv.expand(self.num_Z, 3, -1, -1)
        self.w_z_f = nn.Parameter(self.w_z_f0, requires_grad=True)
        #self.w_z1_f0 = w_x1_conv.expand(self.num_Z, 3, -1, -1)
        #self.w_z1_f = nn.Parameter(self.w_z1_f0, requires_grad=True)
        # proxNet in initialization process
        # 3 means R,G,B channels for color image
        self.xnet = Xnet(self.num_Z + 3)
        self.unet = Unet(self.num_Z + 3)
        self.x1net = X1net(self.num_Z + 3)
        self.m1net = M1net(self.num_Z1)
        self.znet = Znet(self.num_Z1)
        self.z1net = Z1net(self.num_Z1)
        self.z2net = Z2net(self.num_Z1)

        self.downsample = Downsample(self.wavename)
        self.upsample = Upsample(self.wavename)
        # proxNet in iterative process
        self.x_stage = self.make_xnet(self.S, self.num_Z + 3)
        self.u_stage = self.make_unet(self.S, self.num_Z + 3)
        self.x1_stage = self.make_x1net(self.S, self.num_Z + 3)
        self.m1_stage = self.make_m1net(self.S, self.num_Z1)
        self.z_stage = self.make_znet(self.S, self.num_Z1)
        self.z1_stage = self.make_z1net(self.S, self.num_Z1)
        self.z2_stage = self.make_z2net(self.S, self.num_Z + 3)

        # fine-tune at the last
        self.fxnet = Xnet(self.num_Z + 3)
        self.fx1net = X1net(self.num_Z + 3)
        self.fm1net = M1net(self.num_Z1)
        self.funet = Unet(self.num_Z + 3)
        self.fznet = Znet(self.num_Z1)
        self.fz1net = Z1net(self.num_Z1)
        self.fz2net = Z2net(self.num_Z + 3)

        self.f = nn.ReLU(inplace=True)

    def make_xnet(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(Xnet(channels))
        return nn.Sequential(*layers)

    def make_unet(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(Unet(channels))
        return nn.Sequential(*layers)

    def make_x1net(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(X1net(channels))
        return nn.Sequential(*layers)

    def make_m1net(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(M1net(channels))
        return nn.Sequential(*layers)

    def make_znet(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(Znet(channels))
        return nn.Sequential(*layers)

    def make_z1net(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(Z1net(channels))
        return nn.Sequential(*layers)

    def make_z2net(self, iters, channels):
        layers = []
        for i in range(iters):
            layers.append(Z2net(channels))
        return nn.Sequential(*layers)

    def make_eta(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def make_eta1(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def make_weight(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1, -1, -1, -1)
        weight = nn.Parameter(data=const_f, requires_grad=True)
        return weight

    def forward(self, input):
        # save mid-updating results
        ListB = []
        ListCM = []
        ListE = []
        ListU = []
        # initialize B0 and Z0 (M0 =0)
        I = torch.ones(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3])
        I = I.to(device)
        I1 = torch.ones(
            input.shape[0],
            self.num_M,
            input.shape[2],
            input.shape[3])
        I1 = I1.to(device)
        z0 = F.conv2d(input, self.w_z_f, stride=1, padding=1)
        input_ini = torch.cat((input, z0), dim=1)
        out_dual = self.xnet(input_ini)
        CM0 = out_dual[:, :3, :, :]
        Z = out_dual[:, 3:, :, :]

        Z1_lamda = torch.zeros(
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3])
        Z_lamda = torch.zeros(
            input.shape[0],
            self.num_M,
            input.shape[2],
            input.shape[3])
        for i in range(np.size(Z1_lamda, 0)):
            for j in range(np.size(Z1_lamda, 1)):
                Z1_lamda[i, j, :, :] = torch.eye(
                    Z1_lamda.shape[2], Z1_lamda.shape[3])
        for i in range(np.size(Z_lamda, 0)):
            for j in range(np.size(Z_lamda, 1)):
                Z_lamda[i, j, :, :] = torch.eye(
                    Z_lamda.shape[2], Z_lamda.shape[3])
        Z_lamda = Z_lamda.to(device)
        Z1_lamda = Z1_lamda.to(device)
        Z2_lamda = Z1_lamda
        Z1 = Z
        Z2 = Z
        MM = input - CM0
        UU = torch.sigmoid((MM))
        MM = torch.relu(MM)

        z1 = F.conv2d(UU, self.w_z_f, stride=1, padding=1)
        input_ini1 = torch.cat((UU, z1), dim=1)
        out_dualU = self.unet(input_ini1)
        UU = out_dualU[:, :3, :, :]
        Z_u = out_dualU[:, 3:, :, :]

        E = input - CM0 - torch.mul(UU, MM)  # torch.zeros_like(input)#
        z2 = F.conv2d(E, self.w_z_f, stride=1, padding=1)
        input_ini2 = torch.cat((E, z2), dim=1)
        out_dualE = self.x1net(input_ini2)
        E = out_dualE[:, :3, :, :]
        Z_E = out_dualE[:, 3:, :, :]

        U = UU - 1 / self.eta55 * Z1_lamda
        u_dual = U  # -self.eta3/self.eta55
        input_dual1 = torch.cat((u_dual, Z_u), dim=1)
        out_dual1 = self.u_stage[0](input_dual1)
        U = out_dual1[:, :3, :, :]
        Z_u = out_dual1[:, 3:, :, :]

        EB = MM  # - 1/self.eta35*Z_lamda
        FM2 = F.conv_transpose2d(EB, self.weight4 / 25, stride=1, padding=2)
        FM3 = F.conv_transpose2d(
            EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=4, dilation=2)
        FM4 = F.conv_transpose2d(
            EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=6, dilation=3)
        EB1 = torch.cat((FM2, FM3, FM4), dim=1)
        EB1 = EB1 - 1 / self.eta35 * Z_lamda
        FM2 = self.m1_stage[0](EB1)  # [:,:16,:,:])
        #FM3 = self.z_stage[0](EB1[:,16:32,:,:]+FM2)
        #FM4 = self.z1_stage[0](EB1[:,32:48,:,:]+FM2+FM3)
        FM51 = FM2  # +EB1#torch.cat((FM2,FM3,FM4), dim=1)
        B = FM51

        UU = torch.mul(self.eta65 * (input - CM0 - E) +
                       Z2_lamda, MM) + self.eta55 * U + Z1_lamda
        UU1 = self.eta55 * I + self.eta65 * torch.mul(MM, MM)
        UU = UU / UU1
        UU = torch.sigmoid((UU))

        M_M = torch.mul(UU, self.eta65 * (input - CM0 - E) +
                        Z2_lamda)  # + B +self.eta35/2*Z_lamda
        EB = M_M
        FM2 = F.conv_transpose2d(
            EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=2)
        FM3 = F.conv_transpose2d(
            EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=4, dilation=2)
        FM4 = F.conv_transpose2d(
            EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=6, dilation=3)
        FM5 = torch.cat((FM2, FM3, FM4), dim=1)
        FM5 = FM5 + self.eta35 * B + Z_lamda
        FM21 = F.conv_transpose2d(
            UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=2)
        FM31 = F.conv_transpose2d(
            UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=4, dilation=2)
        FM41 = F.conv_transpose2d(
            UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=6, dilation=3)
        FM51 = torch.cat((FM21, FM31, FM41), dim=1)
        FM51 = torch.sigmoid(FM51)
        MM1 = self.eta35 * I1 + self.eta65 * torch.mul(FM51, FM51)
        MM2 = FM5 / MM1
        FM22 = F.conv2d(MM2[:, :16, :, :], self.conv4[1, :,
                                                      :, :, :] / 25, stride=1, padding=2)
        FM32 = F.conv2d(MM2[:, 16:32, :, :], self.conv4[1,
                                                        :, :, :, :] / 25, stride=1, padding=4, dilation=2)
        FM42 = F.conv2d(MM2[:, 32:48, :, :], self.conv4[1,
                                                        :, :, :, :] / 25, stride=1, padding=6, dilation=3)
        MM = self.eta4 * FM22 + self.eta5 * FM32 + self.eta6 * FM42
        #FM51 = torch.cat((FM22,FM32,FM42), dim=1)
        #MM = F.conv2d(FM51, self.weight5/1, stride=1,padding = 0)
        MM = torch.relu(MM - self.eta10)  # -self.eta10

        WE = input - torch.mul(UU, MM) - CM0 + 1 / self.eta65 * Z2_lamda
        E_dual = WE  # -self.eta45/self.eta65
        input_dualE = torch.cat((E_dual, Z_E), dim=1)
        out_dualE = self.x1_stage[0](input_dualE)
        E = out_dualE[:, :3, :, :]
        Z_E = out_dualE[:, 3:, :, :]

        ES = input - E - torch.mul(UU, MM) + 1 / self.eta65 * Z2_lamda
        ECM = ES  # -self.eta1/self.eta65#self.f(ES + self.tau)
        FM = torch.cat((ECM, Z), dim=1)
        out_dualx = self.x_stage[0](FM)
        CM = out_dualx[:, :3, :, :]
        Z = out_dualx[:, 3:, :, :]

        Z2_lamda = Z2_lamda + self.eta65 * (input - torch.mul(UU, MM) - CM - E)
        Z_lamda = Z_lamda + self.eta35 * (B - MM2)
        Z1_lamda = Z1_lamda + self.eta55 * (U - UU)

        ListE.append(E)
        ListB.append(MM)
        ListU.append(UU)
        ListCM.append(CM)
        for i in range(self.iter):

            U = UU - 1 / self.eta120[i, :] * Z1_lamda
            u_dual = U  # -self.eta13[i,:]/self.eta120[i,:]
            input_dual1 = torch.cat((u_dual, Z_u), dim=1)
            out_dual1 = self.u_stage[i + 1](input_dual1)
            U = out_dual1[:, :3, :, :]
            Z_u = out_dual1[:, 3:, :, :]

            FM5 = MM2 - 1 / self.eta117[i, :] * Z_lamda
            x_dual = FM5  # - self.eta12[i,:]/self.eta117[i,:]
            FM2 = self.m1_stage[i + 1](x_dual)  # [:,:16,:,:])
            #FM3 = self.z_stage[i+1](x_dual[:,16:32,:,:]+FM2)
            #FM4 = self.z1_stage[i+1](x_dual[:,32:48,:,:]+FM2+FM3)
            FM5 = FM2  # +x_dual #torch.cat((FM2,FM3,FM4), dim=1)
            B = FM5

            UU = torch.mul(self.eta121[i,
                                       :] * (input - CM - E) + Z2_lamda,
                           MM) + self.eta120[i,
                                             :] * U + Z1_lamda
            UU1 = self.eta120[i, :] * I + self.eta121[i, :] * torch.mul(MM, MM)
            UU = UU / UU1
            UU = torch.sigmoid((UU))

            # + B +self.eta117[i,:]/2*Z_lamda
            M_M = torch.mul(UU, self.eta121[i, :]
                            * (input - CM - E) + Z2_lamda)
            EB = M_M
            FM2 = F.conv_transpose2d(
                EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=2)
            FM3 = F.conv_transpose2d(
                EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=4, dilation=2)
            FM4 = F.conv_transpose2d(
                EB, self.conv4[1, :, :, :, :] / 25, stride=1, padding=6, dilation=3)
            FM5 = torch.cat((FM2, FM3, FM4), dim=1)
            FM5 = FM5 + self.eta117[i, :] * B + Z_lamda
            FM21 = F.conv_transpose2d(
                UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=2)
            FM31 = F.conv_transpose2d(
                UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=4, dilation=2)
            FM41 = F.conv_transpose2d(
                UU, self.conv4[1, :, :, :, :] / 25, stride=1, padding=6, dilation=3)
            FM51 = torch.cat((FM21, FM31, FM41), dim=1)
            FM51 = torch.sigmoid(FM51)
            MM1 = self.eta117[i, :] * I1 + \
                self.eta121[i, :] * torch.mul(FM51, FM51)
            MM2 = FM5 / MM1
            FM22 = F.conv2d(MM2[:, :16, :, :], self.conv4[1,
                                                          :, :, :, :] / 25, stride=1, padding=2)
            FM32 = F.conv2d(MM2[:, 16:32, :, :], self.conv4[1,
                                                            :, :, :, :] / 25, stride=1, padding=4, dilation=2)
            FM42 = F.conv2d(MM2[:, 32:48, :, :], self.conv4[1,
                                                            :, :, :, :] / 25, stride=1, padding=6, dilation=3)
            MM = self.eta14[i, :] * FM22 + self.eta15[i, :] * \
                FM32 + self.eta16[i, :] * FM42  # +self.eta17[i,:]*MM
            #FM51 = torch.cat((FM22,FM32,FM42), dim=1)
            #MM = F.conv2d(FM51, self.conv5[1,:,:,:,:]/1, stride=1,padding = 0)
            MM = torch.relu(MM - self.eta110[i, :])  # -self.eta110[i,:]

            WE = input - torch.mul(UU, MM) - CM + 1 / \
                self.eta121[i, :] * Z2_lamda
            E_dual = WE  # -self.eta112[i,:]/self.eta121[i,:]
            input_dualE = torch.cat((E_dual, Z_E), dim=1)
            out_dualE = self.x1_stage[i + 1](input_dualE)
            E = out_dualE[:, :3, :, :]
            Z_E = out_dualE[:, 3:, :, :]

            # M-net
            ES = input - E - torch.mul(UU, MM) + 1 / \
                self.eta121[i, :] * Z2_lamda
            # -self.eta11[i,:]/self.eta121[i,:]#self.f(ES + self.tau)#
            ECM = ES
            FM = torch.cat((ECM, Z), dim=1)
            out_dualx = self.x_stage[i + 1](FM)
            CM = out_dualx[:, :3, :, :]
            Z = out_dualx[:, 3:, :, :]

            Z2_lamda = Z2_lamda + \
                self.eta121[i, :] * (input - torch.mul(UU, MM) - CM - E)
            Z_lamda = Z_lamda + self.eta117[i, :] * (B - MM2)
            Z1_lamda = Z1_lamda + self.eta120[i, :] * (U - UU)

            ListB.append(MM)
            ListU.append(UU)
            ListE.append(E)
            ListCM.append(CM)

        # out_dualE = self.fx1net(out_dualE)                # fine-tune
        #E = out_dualE[:,:3,:,:]
        out_dualx = self.fxnet(out_dualx)
        CM = out_dualx[:, :3, :, :]

        ListCM.append(CM)
        # ListE.append(E)
        return CM0, ListB, ListCM, ListU, ListE

# proxNet_M


class M1net(nn.Module):
    def __init__(self, channels):
        super(M1net, self).__init__()
        self.channels = channels
        self.tau0 = torch.Tensor([0.5])
        self.taum = self.tau0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau = nn.Parameter(self.taum, requires_grad=True)
        self.f = nn.ReLU(inplace=True)
        self.resm1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resm8 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        m1 = F.relu(input + self.resm1(input))
        m2 = F.relu(m1 + self.resm2(m1))
        m3 = F.relu(m2 + self.resm3(m2))
        m4 = F.relu(m3 + self.resm4(m3))
        # for sparse rain map
        m_rev = self.f(m4 - self.tau)
        return m_rev

# proxNet_B


class Xnet(nn.Module):
    def __init__(self, channels):
        super(Xnet, self).__init__()
        self.channels = channels
        self.taux0 = torch.Tensor([0.1])
        self.taux = self.taux0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau1 = nn.Parameter(self.taux, requires_grad=True)
        self.f = nn.ReLU(inplace=True)
        self.resx1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx8 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        x3 = F.relu(x2 + self.resx3(x2))
        x4 = F.relu(x3 + self.resx4(x3))
        #x5 = F.relu(x4+self.resx5(x4))
        #x6 = F.relu(x5+self.resx6(x5))
        #x7 = F.relu(x6+self.resx7(x6))
        #x8 = F.relu(self.resx8(x7))
        #x_rev =self.f(x5-self.tau1)
        return x4


class X1net(nn.Module):
    def __init__(self, channels):
        super(X1net, self).__init__()
        self.channels = channels
        self.tau0 = torch.Tensor([0.5])
        self.tauw = self.tau0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau = nn.Parameter(self.tauw, requires_grad=True)
        self.f = nn.ReLU(inplace=True)
        self.resx1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resx8 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        x1 = F.relu(input + self.resx1(input))
        x2 = F.relu(x1 + self.resx2(x1))
        x3 = F.relu(x2 + self.resx3(x2))
        x4 = F.relu(x3 + self.resx4(x3))
        #x5 = F.relu(x4+self.resx5(x4))
        #x6 = F.relu(x5+self.resx6(x5))
        #x7 = F.relu(x6+self.resx7(x6))
        #x8 = F.relu(self.resx8(x7))
        #x_rev =self.f(x4-self.tau)
        return x4


# proxNet_U
class Unet(nn.Module):
    def __init__(self, channels):
        super(Unet, self).__init__()
        self.channels = channels
        self.tauu0 = torch.Tensor([0.5])
        self.tauu = self.tauu0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau11 = nn.Parameter(self.tauu, requires_grad=True)
        self.f = torch.nn.Sigmoid()
        self.resu1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=3,
                dilation=3),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=4,
                dilation=4),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu8 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resu7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        u1 = F.relu(input + self.resu1(input))
        u2 = F.relu(u1 + self.resu2(u1))
        u3 = F.relu(u2 + self.resu3(u2))
        u4 = self.f((u3 + self.resu4(u3)))
        #u5  = self.f(u4+self.resu5(u4))
        #u6 = self.f(u5+self.resu6(u5))
        #u7 =F.relu((u4-self.tau11))
        #u_rev =(self.f(u7)-0.5)*2
        return u4


class Znet(nn.Module):
    def __init__(self, channels):
        super(Znet, self).__init__()
        self.channels = channels
        self.tauu0 = torch.Tensor([0.1])
        self.tauu = self.tauu0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau11 = nn.Parameter(self.tauu, requires_grad=True)
        self.f = torch.nn.Sigmoid()
        self.resz1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        z1 = F.relu(input + self.resz1(input))
        z2 = F.relu(z1 + self.resz2(z1))
        z3 = F.relu(z2 + self.resz3(z2))
        z4 = F.relu((z3 + self.resz4(z3)))
        z_rev = self.f(z4 - self.tau11)
        return z_rev


class Z1net(nn.Module):
    def __init__(self, channels):
        super(Z1net, self).__init__()
        self.channels = channels
        self.tauu0 = torch.Tensor([0.1])
        self.tauu = self.tauu0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau11 = nn.Parameter(self.tauu, requires_grad=True)
        self.f = torch.nn.Sigmoid()
        self.resz1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        z1 = F.relu(input + self.resz1(input))
        z2 = F.relu(z1 + self.resz2(z1))
        z3 = F.relu(z2 + self.resz3(z2))
        z4 = F.relu((z3 + self.resz4(z3)))
        z_rev = self.f(z4 - self.tau11)
        return z_rev


class Z2net(nn.Module):
    def __init__(self, channels):
        super(Z2net, self).__init__()
        self.channels = channels
        self.tauu0 = torch.Tensor([0.1])
        self.tauu = self.tauu0.unsqueeze(dim=0).unsqueeze(
            dim=0).unsqueeze(dim=0).expand(-1, self.channels, -1, -1)
        # for sparse rain map
        self.tau11 = nn.Parameter(self.tauu, requires_grad=True)
        self.f = torch.nn.Sigmoid()
        self.resz1 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz2 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz3 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz4 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz5 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz6 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )
        self.resz7 = nn.Sequential(
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
            nn.ReLU(),
            nn.Conv2d(
                self.channels,
                self.channels,
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1),
            nn.BatchNorm2d(
                self.channels),
        )

    def forward(self, input):
        z1 = F.relu(input + self.resz1(input))
        z2 = F.relu(z1 + self.resz2(z1))
        return z2


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


class Downsample(nn.Module):
    def __init__(self, wavename='haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input):
        LL, LR, RL, RR = self.dwt(input)
        return LL, LR, RL, RR


class Upsample(nn.Module):
    def __init__(self, wavename='haar'):
        super(Upsample, self).__init__()
        self.dwt = IDWT_2D(wavename=wavename)

    def forward(self, LL, LR, RL, RR):
        output = self.dwt(LL, LR, RL, RR)
        return output


class DWT_2D(nn.Module):

    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height -
                                 math.floor(self.input_height /
                                            2)), 0:(self.input_height +
                                                    self.band_length -
                                                    2)]
        matrix_g_1 = matrix_g[0:(self.input_width -
                                 math.floor(self.input_width /
                                            2)), 0:(self.input_width +
                                                    self.band_length -
                                                    2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):

        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(
            input,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1)


class IDWT_2D(nn.Module):

    def __init__(self, wavename):

        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):

        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            -self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height -
                                 math.floor(self.input_height /
                                            2)), 0:(self.input_height +
                                                    self.band_length -
                                                    2)]
        matrix_g_1 = matrix_g[0:(self.input_width -
                                 math.floor(self.input_width /
                                            2)), 0:(self.input_width +
                                                    self.band_length -
                                                    2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):

        assert len(
            LL.size()) == len(
            LH.size()) == len(
            HL.size()) == len(
                HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(
            LL,
            LH,
            HL,
            HH,
            self.matrix_low_0,
            self.matrix_low_1,
            self.matrix_high_0,
            self.matrix_high_1)


class DWTFunction_2D(Function):
    @staticmethod
    def forward(
            ctx,
            input,
            matrix_Low_0,
            matrix_Low_1,
            matrix_High_0,
            matrix_High_1):
        ctx.save_for_backward(
            matrix_Low_0,
            matrix_Low_1,
            matrix_High_0,
            matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH

    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(
            torch.matmul(
                grad_LL, matrix_Low_1.t()), torch.matmul(
                grad_LH, matrix_High_1.t()))
        grad_H = torch.add(
            torch.matmul(
                grad_HL, matrix_Low_1.t()), torch.matmul(
                grad_HH, matrix_High_1.t()))
        grad_input = torch.add(
            torch.matmul(
                matrix_Low_0.t(), grad_L), torch.matmul(
                matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class IDWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input_LL, input_LH, input_HL, input_HH,
                matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(
            matrix_Low_0,
            matrix_Low_1,
            matrix_High_0,
            matrix_High_1)
        L = torch.add(
            torch.matmul(
                input_LL, matrix_Low_1.t()), torch.matmul(
                input_LH, matrix_High_1.t()))
        H = torch.add(
            torch.matmul(
                input_HL, matrix_Low_1.t()), torch.matmul(
                input_HH, matrix_High_1.t()))
        output = torch.add(
            torch.matmul(
                matrix_Low_0.t(), L), torch.matmul(
                matrix_High_0.t(), H))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1)
        grad_LH = torch.matmul(grad_L, matrix_High_1)
        grad_HL = torch.matmul(grad_H, matrix_Low_1)
        grad_HH = torch.matmul(grad_H, matrix_High_1)
        return grad_LL, grad_LH, grad_HL, grad_HH, None, None, None, None
