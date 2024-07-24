import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as torch_fun
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
import os

from TumorDetection.utils.dict_classes import DataPathDir, ReportingPathDir, Device
from TumorDetection.data.loader import DataPathLoader
from TumorDetection.data.dataset import TorchDatasetSeg
from TumorDetection.models.efsnet import EFSNetSeg
from TumorDetection.models.adv_semi_seg.discriminator import FCDiscriminator
from TumorDetection.models.adv_semi_seg.losses import CrossEntropy2d, BCEWithLogitsLoss2d

DEVICE = Device.device
TEST_SIZE = 100

MODEL_NAME = 'EFSNet_ADVSS'
INPUT_SIZE = (256, 256)
NUM_CLASSES = 3
BATCH_SIZE = 16
ITER_SIZE = 1
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 20000
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_PRED = 0.1

PARTIAL_DATA = 0.5

SEMI_START = 5000
LAMBDA_SEMI = 0.1
MASK_T = 0.2

LAMBDA_SEMI_ADV = 0.001
SEMI_START_ADV = 0
D_REMAIN = True


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long())
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter_, max_iter, power):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_d(optimizer, i_iter):
    lr = lr_poly(LEARNING_RATE_D, i_iter, NUM_STEPS, POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def one_hot(label):
    label = label.numpy()
    onehot = np.zeros((label.shape[0], NUM_CLASSES, label.shape[1], label.shape[2]),
                      dtype=label.dtype)
    for i in range(NUM_CLASSES):
        onehot[:, i, ...] = (label == i)
    # handle ignore labels
    return torch.FloatTensor(onehot)


def make_d_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    d_label = np.ones(ignore_mask.shape) * label
    d_label[ignore_mask] = 255
    d_label = Variable(torch.FloatTensor(d_label))
    return d_label


def main():
    h, w = map(int, INPUT_SIZE)
    input_size = (h, w)

    cudnn.enabled = True
    gpu = 0

    # create network
    model = EFSNetSeg(input_shape=(1, h, w),
                      device=DEVICE)

    model.train()
    model.cuda(gpu)
    cudnn.benchmark = True

    # init D
    discr = FCDiscriminator(num_classes=model.num_classes)
    discr.train()
    discr.cuda(gpu)

    dp = DataPathLoader(DataPathDir.get('dir_path'))
    paths = dp()
    tr_paths, _ = train_test_split(paths, test_size=TEST_SIZE,
                                   random_state=0, shuffle=True)
    tr_td = TorchDatasetSeg(tr_paths,
                            crop_prob=0.5,
                            rotation_degrees=180,
                            range_contrast=(0.75, 1.25),
                            range_brightness=(0.75, 1.25),
                            vertical_flip_prob=0.25,
                            horizontal_flip_prob=0.25)
    tr_gt_td = TorchDatasetSeg(tr_paths,
                               crop_prob=0.5,
                               rotation_degrees=180,
                               range_contrast=(0.75, 1.25),
                               range_brightness=(0.75, 1.25),
                               vertical_flip_prob=0.25,
                               horizontal_flip_prob=0.25)

    partial_size = int(PARTIAL_DATA * len(tr_td))
    train_ids = list(range(len(tr_td)))
    np.random.shuffle(train_ids)
    train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
    train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
    train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

    trainloader = data.DataLoader(tr_td,
                                  batch_size=BATCH_SIZE,
                                  sampler=train_sampler)
    trainloader_remain = data.DataLoader(tr_td,
                                         batch_size=BATCH_SIZE,
                                         sampler=train_remain_sampler)
    trainloader_gt = data.DataLoader(tr_gt_td,
                                     batch_size=BATCH_SIZE,
                                     sampler=train_gt_sampler)

    trainloader_remain_iter = enumerate(trainloader_remain)
    trainloader_iter = enumerate(trainloader)
    trainloader_gt_iter = enumerate(trainloader_gt)

    # implement model.optim_parameters(args) to handle different models' lr setting

    # optimizer for segmentation network
    optimizer = optim.SGD(model.parameters(),
                          lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    optimizer.zero_grad()

    # optimizer for discriminator network
    optimizer_d = optim.Adam(discr.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_d.zero_grad()

    # loss/ bilinear upsampling
    bce_loss = BCEWithLogitsLoss2d()
    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    pred_label = 0
    gt_label = 1

    for i_iter in range(NUM_STEPS):

        loss_seg_value = 0
        loss_adv_pred_value = 0
        loss_d_value = 0
        loss_semi_value = 0
        loss_semi_adv_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        optimizer_d.zero_grad()
        adjust_learning_rate_d(optimizer_d, i_iter)

        for sub_i in range(ITER_SIZE):
            # train G
            # don't accumulate grads in D
            for param in discr.parameters():
                param.requires_grad = False

            # do semi first
            if (LAMBDA_SEMI > 0 or LAMBDA_SEMI_ADV > 0) and i_iter >= SEMI_START_ADV:
                try:
                    _, batch = next(trainloader_remain_iter)
                except StopIteration:
                    trainloader_remain_iter = enumerate(trainloader_remain)
                    _, batch = next(trainloader_remain_iter)

                # only access to img
                images, _ = batch
                images = Variable(images.to(DEVICE))

                pred = interp(model(images))
                pred_remain = pred.detach()

                d_out = interp(discr(torch_fun.softmax(pred, dim=1).to(DEVICE)))
                d_out_sigmoid = torch_fun.sigmoid(d_out).data.cpu().numpy().squeeze(axis=1)

                ignore_mask_remain = np.zeros(d_out_sigmoid.shape).astype(bool)
                # noinspection PyUnresolvedReferences
                loss_semi_adv = LAMBDA_SEMI_ADV * bce_loss(d_out,
                                                           make_d_label(gt_label,
                                                                        ignore_mask_remain).to(DEVICE))
                loss_semi_adv = loss_semi_adv / ITER_SIZE
                loss_semi_adv_value += loss_semi_adv.data.cpu().numpy() / LAMBDA_SEMI_ADV

                if LAMBDA_SEMI <= 0 or i_iter < SEMI_START:
                    loss_semi_adv.backward()
                    loss_semi_value = 0
                else:
                    # produce ignore mask
                    semi_ignore_mask = (d_out_sigmoid < MASK_T)

                    semi_gt = pred.data.cpu().numpy().argmax(axis=1)
                    semi_gt[semi_ignore_mask] = 255

                    semi_ratio = 1.0 - float(semi_ignore_mask.sum()) / semi_ignore_mask.size
                    print('semi ratio: {:.4f}'.format(semi_ratio))

                    if semi_ratio == 0.0:
                        loss_semi_value += 0
                    else:
                        semi_gt = torch.FloatTensor(semi_gt)

                        loss_semi = LAMBDA_SEMI * loss_calc(pred, semi_gt, gpu)
                        loss_semi = loss_semi / ITER_SIZE
                        loss_semi_value += loss_semi.data.cpu().numpy()[0] / LAMBDA_SEMI
                        loss_semi += loss_semi_adv
                        loss_semi.backward()

            # else:
            #     loss_semi = None
            #     loss_semi_adv = None

            # train with source

            try:
                _, batch = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = enumerate(trainloader)
                _, batch = next(trainloader_iter)

            images, labels = batch
            images = Variable(images.to(DEVICE))
            ignore_mask = (labels.numpy() == 255)
            pred = interp(model(images))

            loss_seg = loss_calc(pred, labels, gpu)

            d_out = interp(discr(torch_fun.softmax(pred, dim=1)))

            loss_adv_pred = bce_loss(d_out, make_d_label(gt_label, ignore_mask))

            loss = loss_seg + LAMBDA_ADV_PRED * loss_adv_pred

            # proper normalization
            loss = loss / ITER_SIZE
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy() / ITER_SIZE
            loss_adv_pred_value += loss_adv_pred.data.cpu().numpy() / ITER_SIZE

            # train D

            # bring back requires_grad
            for param in discr.parameters():
                param.requires_grad = True

            # train with pred
            pred = pred.detach()

            if D_REMAIN:
                # noinspection PyUnboundLocalVariable
                pred = torch.cat((pred, pred_remain), 0)
                # noinspection PyUnboundLocalVariable
                ignore_mask = np.concatenate((ignore_mask, ignore_mask_remain), axis=0)

            d_out = interp(discr(torch_fun.softmax(pred, dim=1)))
            loss_d = bce_loss(d_out, make_d_label(pred_label, ignore_mask))
            loss_d = loss_d / ITER_SIZE / 2
            loss_d.backward()
            loss_d_value += loss_d.data.cpu().numpy()

            # train with gt
            # get gt labels
            try:
                _, batch = next(trainloader_gt_iter)
            except StopIteration:
                trainloader_gt_iter = enumerate(trainloader_gt)
                _, batch = next(trainloader_gt_iter)

            _, labels_gt = batch
            d_gt_v = Variable(one_hot(labels_gt).to(DEVICE))
            ignore_mask_gt = (labels_gt.numpy() == 255)

            d_out = interp(discr(d_gt_v))
            # noinspection PyUnresolvedReferences
            loss_d = bce_loss(d_out, make_d_label(gt_label, ignore_mask_gt).to(DEVICE))
            loss_d = loss_d / ITER_SIZE / 2
            loss_d.backward()
            loss_d_value += loss_d.data.cpu().numpy()

        optimizer.step()
        optimizer_d.step()

        print(
            'iter = {0:8d}/{1:8d},'
            ' loss_seg = {2:.3f},'
            ' loss_adv_p = {3:.3f}, '
            'loss_D = {4:.3f}, '
            'loss_semi = {5:.3f}, '
            'loss_semi_adv = {6:.3f}'.format(
                i_iter, NUM_STEPS, loss_seg_value, loss_adv_pred_value,
                loss_d_value, loss_semi_value,
                loss_semi_adv_value))

        if i_iter >= NUM_STEPS - 1:
            print('save model ...')
            torch.save(model.state_dict(), os.path.join(ReportingPathDir.dir_path, 'adv_ss',
                                                        MODEL_NAME + '.pth'))
            torch.save(discr.state_dict(), os.path.join(ReportingPathDir.dir_path, 'adv_ss',
                                                        MODEL_NAME + '_discr.pth'))
            break


if __name__ == '__main__':
    main()
