import os
import math
from decimal import Decimal
import utility
import IPython
import torch
import re
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio 
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import torch.nn as nn
import torchvision
from cal_ssim import SSIM
#from skimage.measure import compare_ssim
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # 保证程序中的GPU序号是和硬件中的序号是相同的
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 使用几块GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.S = args.stage
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()
            for _ in range(len(ckp.ssim_log)): self.scheduler.step()
        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        self.ssim = SSIM().to(device)
        timer_data, timer_model = utility.timer(), utility.timer()

        loss_Bs_all = 0
        loss_Rs_all = 0
        loss_Us_all = 0
        loss_Ls_all = 0
        loss_Hs_all = 0
        loss_B_all = 0
        loss_R_all=0
        loss_U_all = 0
        loss_L_all = 0
        loss_H_all = 0
        cnt = 0
        for batch, (lr, hr, idx_scale) in enumerate(self.loader_train):
            loss_Bs = 0
            loss_Rs = 0
            loss_Us = 0
            loss_Ls = 0
            loss_Hs = 0
            cnt = cnt+1
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            I = torch.ones(lr.shape[0], lr.shape[1], lr.shape[2], lr.shape[3])
            I = I.to(device)
            B0, ListB, ListR,ListU,ListE = self.model(lr, idx_scale)
            #B0, ListB, ListR, ListE = self.model(lr, idx_scale)
            
            #l_h = torch.sigmoid(lr-hr)
            #LL = (torch.sigmoid(lr-hr)-0.5)*2
            l_h = lr-hr
            LL1 = l_h.cpu().numpy()
            LL1 = np.where(LL1>0, 1, 0)
            LL = torch.from_numpy(LL1)
            LL = LL.type(torch.FloatTensor)
            LL = LL.to(device)
            self.f = nn.L1Loss()
            for j in range(self.S):
                loss_Bs = float(loss_Bs) + 0.1*self.loss(ListR[j], hr)
                loss_Rs = float(loss_Rs) + 0.1*self.loss(torch.mul(ListU[j],ListB[j])+ListE[j],lr-hr)
                loss_Ls = float(loss_Ls) + 0.1*self.f(ListU[j], LL)
                loss_Us = float(loss_Us) + 0.1*(1-self.ssim(ListR[j], hr))
                loss_Hs = float(loss_Hs) + 0.1*(1-self.ssim(torch.mul(ListU[j],ListB[j])+ListE[j],lr-hr))
            loss_B = self.loss(ListR[-1], hr)
            loss_R = 0.9*self.loss(torch.mul(ListU[-1],ListB[-1])+ListE[-1],lr-hr)
            loss_L = 0.9*self.f(ListU[-1], LL)
            loss_U = 1*(1-self.ssim(ListR[-1], hr))
            loss_H = 0.9*(1-self.ssim(torch.mul(ListU[-1],ListB[-1])+ListE[-1],lr-hr))             
            loss_B0 = 0.1* self.loss(B0, hr)
            loss = loss_B + loss_Bs + loss_Rs + loss_R+loss_B0+loss_Ls+loss_L+loss_Us+loss_U+loss_Hs+loss_H#
            loss_Bs_all = loss_Bs_all + loss_Bs
            loss_B_all = loss_B_all + loss_B
            loss_Rs_all = loss_Rs_all + loss_Rs
            loss_R_all = loss_R_all + loss_R
            loss_L_all = loss_L_all + loss_L
            loss_Ls_all = loss_Ls_all + loss_Ls
            loss_U_all = loss_U_all + loss_U
            loss_Us_all = loss_Us_all + loss_Us
            loss_H_all = loss_H_all + loss_H
            loss_Hs_all = loss_Hs_all + loss_Hs
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                ttt = 0
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        print(loss_Bs_all/cnt)
        print(loss_B_all / cnt)
        print(loss_Rs_all / cnt)
        print(loss_R_all / cnt)
        print(loss_Ls_all / cnt)
        print(loss_L_all / cnt)
        print(loss_Us_all / cnt)
        print(loss_U_all / cnt)
        print(loss_Hs_all / cnt)
        print(loss_H_all / cnt)
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                #ssim1 = 0
                #ssim = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)
                    B0, ListB,ListR,ListU,ListE= self.model(lr, idx_scale)
                    sr = utility.quantize(ListR[-1], self.args.rgb_range)    # restored background at the last stage
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    #if self.args.save_results:
                    self.ckp.save_results(filename, save_list, scale)
                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                #ssim = ssim1 / len(self.loader_test)
                #print(ssim)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )
        
        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs