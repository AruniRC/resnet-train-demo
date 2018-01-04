import datetime
import math
import os
import os.path as osp
import shutil

# import fcn
import numpy as np
import PIL.Image
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import utils
# import torchfcn
import gc



# -----------------------------------------------------------------------------
def cross_entropy2d(input, target, weight=None, size_average=True):
# -----------------------------------------------------------------------------
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


# -----------------------------------------------------------------------------
def kl_div2d(input, target, weight=None, size_average=True):
# -----------------------------------------------------------------------------
    # n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    target = target.permute(0,3,1,2) # 1 x num_bins x H x W
    assert log_p.size() == target.size()
    loss = F.kl_div(log_p, target, size_average=True)
    return loss


# -----------------------------------------------------------------------------
def mse2d(input, target, weight=None, size_average=True):
# -----------------------------------------------------------------------------
    # n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    target = target.permute(0,3,1,2) # 1 x num_bins x H x W
    input = F.softmax(input)    
    assert input.size() == target.size()
    loss = F.mse_loss(input, target)
    return loss



class Trainer(object):

    # -----------------------------------------------------------------------------
    def __init__(self, cuda, model, criterion, optimizer,
                 train_loader, val_loader, out, max_iter, interval_validate=None):
    # -----------------------------------------------------------------------------
        self.cuda = cuda

        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('US/Eastern'))

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'valid/loss',
            'valid/acc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_loss = 0


    # -----------------------------------------------------------------------------
    def validate(self):
    # -----------------------------------------------------------------------------
        training = self.model.training
        self.model.eval()
        MAX_NUM = 500 # HACK: stop after 500 images

        n_class = self.val_loader.dataset.classses
        val_loss = 0
        label_trues, label_preds = [], []

        for batch_idx, (data, (target)) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=True):

            # Computing val losses
            if self.train_loader.dataset.bins == 'one-hot':
                target_hue, target_chroma = target
                if self.cuda:
                    data, target_hue, target_chroma = \
                        data.cuda(), target_hue.cuda(), target_chroma.cuda()
                data, target_hue, target_chroma = \
                    Variable(data), Variable(target_hue), Variable(target_chroma)
                (score_hue, score_chroma) = self.model(data)

                loss_hue = cross_entropy2d(score_hue, target_hue,
                               size_average=self.size_average)
                loss_chroma = cross_entropy2d(score_chroma, target_chroma,
                               size_average=self.size_average)
                if np.isnan(float(loss_hue.data[0])):
                    raise ValueError('hue loss is NaN while validation')
                if np.isnan(float(loss_chroma.data[0])):
                    raise ValueError('chroma loss is NaN while validation')

                loss = loss_chroma + loss_hue
                val_loss += float(loss.data[0]) / len(data)
                del loss_hue, loss_chroma

            elif self.train_loader.dataset.bins == 'soft':
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                score = self.model(data)
                loss = kl_div2d(score, target, size_average=self.size_average) # DEBUG: MSE loss
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is NaN while validation')
                val_loss += float(loss.data[0]) / len(data)

            # Visualization
            imgs = data.data.cpu()
            if self.train_loader.dataset.bins == 'one-hot':
                # visualize only hue predictions
                lbl_pred = score_hue.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target_hue.data.cpu()
                del score_hue, target_hue, score_chroma, target_chroma
            elif self.train_loader.dataset.bins == 'soft':
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                _, lbl_true = target.data.max(dim=3) # EDIT - .data
                lbl_true = lbl_true.cpu()
                del target, score

            lbl_pred = lbl_pred.squeeze()
            lbl_true = np.squeeze(lbl_true.numpy())
            
            if len(visualizations) < 9:
                assert imgs.size()[0]==1   # HACK: assumes 1 image in a batch!
                img = \
                    PIL.Image.open(self.val_loader.dataset.files['val'][batch_idx]) 
                img = self.val_loader.dataset.rescale(img) # orig RGB image
                img_l = np.squeeze(imgs.numpy()) + self.val_loader.dataset.mean_l
                viz = utils.visualize_segmentation(lbl_pred=lbl_pred,
                        lbl_true=lbl_true, img=img, 
                        im_l=img_l, n_class=n_class)
                visualizations.append(viz)

            label_trues.append(lbl_true)
            label_preds.append(lbl_pred)

            del lbl_true, lbl_pred, data, loss, imgs

            if batch_idx > MAX_NUM:
                break

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, utils.get_tile_image(visualizations))
        del visualizations

        # Computing metrics
        metrics = utils.label_accuracy_score(
            label_trues, label_preds, n_class)
        val_loss /= len(self.val_loader)
        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('US/Eastern')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        del label_trues, label_preds, val_loss
        gc.collect()

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()



    # -----------------------------------------------------------------------------
    def train_epoch(self):
    # -----------------------------------------------------------------------------
        self.model.train()
        n_class = len(self.train_loader.dataset.classes)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)

            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            # if self.iteration % self.interval_validate == 0:
            #     self.validate()

            assert self.model.training

            # Computing Losses
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            score = self.model(data)

            loss = self.criterion(score, target)

            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is NaN while training')
            # print list(self.model.parameters())[0].grad

            # Gradient descent
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # Computing metrics
            # TODO

            # Logging
            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                # elapsed_time = (
                #     datetime.datetime.now(pytz.timezone('US/Eastern')) -
                #     self.timestamp_start).total_seconds()
                # log = [self.epoch, self.iteration] + [loss.data[0]] + \
                #     metrics.tolist() + [''] * 5 + [elapsed_time]
                # log = map(str, log)
                # f.write(','.join(log) + '\n')
                print '\nEpoch: ' + str(self.epoch) + ' Iter: ' + str(self.iteration) + \
                        ' Loss: ' + str(loss.data[0])

            if self.iteration >= self.max_iter:
                break


    # -----------------------------------------------------------------------------
    def eval_metric(self, score, target, n_class):
    # -----------------------------------------------------------------------------
        metrics = []
        lbl_pred = score.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.cpu().numpy()
        for lt, lp in zip(lbl_true, lbl_pred):
            acc, acc_cls, mean_iu, fwavacc = \
                utils.label_accuracy_score(
                    [lt], [lp], n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
        metrics = np.mean(metrics, axis=0)
        return metrics


    # -----------------------------------------------------------------------------
    def train(self):
    # -----------------------------------------------------------------------------
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break