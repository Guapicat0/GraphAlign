import os
import matplotlib
import torch
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
from torch.utils.tensorboard import SummaryWriter



class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []

        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")



import numpy as np

class EvalHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir        = log_dir
        self.best_avg_metric = -np.inf
        self.best_krocc     = -np.inf
        self.best_srocc     = -np.inf
        self.best_plcc      = -np.inf

        self.krocc          = []
        self.srocc          = []
        self.plcc           = []

        os.makedirs(self.log_dir, exist_ok=True)
        self.writer         = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_eval(self, epoch, krocc, srocc, plcc):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.krocc.append(krocc)
        self.srocc.append(srocc)
        self.plcc.append(plcc)

        self.writer.add_scalar('krocc', krocc, epoch)
        self.writer.add_scalar('srocc', srocc, epoch)
        self.writer.add_scalar('plcc', plcc, epoch)

        self.update_best_metrics(krocc, srocc, plcc)

    def update_best_metrics(self, krocc, srocc, plcc):
        avg_metric = (krocc + srocc + plcc) / 3
        if avg_metric > self.best_avg_metric:
            self.best_avg_metric = avg_metric
            self.best_krocc = krocc
            self.best_srocc = srocc
            self.best_plcc = plcc

    def plot_eval(self):
        iters = range(len(self.krocc))

        plt.figure()
        plt.plot(iters, self.krocc, 'blue', linewidth=2, label='KROCC')
        plt.plot(iters, self.srocc, 'green', linewidth=2, label='SROCC')
        plt.plot(iters, self.plcc, 'red', linewidth=2, label='PLCC')

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_eval_metrics.png"))

        plt.cla()
        plt.close("all")