import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from model import ENet
from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH + '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()


def main():

    cfg_file = open('./config.py', "r")
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID) == 1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.TRAIN.STAGE == 'all':
        net = ENet(only_encode=False)  # ENet / BiSeNet / ICNet
        if cfg.TRAIN.PRETRAINED_ENCODER != '':
            # Load the pretrained_encoder
            encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
            del encoder_weight['classifier.bias']
            del encoder_weight['classifier.weight']
            # pdb.set_trace()
            net.encoder.load_state_dict(encoder_weight)
    elif cfg.TRAIN.STAGE == 'encoder':
        net = ENet(only_encode=True)

    if len(cfg.TRAIN.GPU_ID) > 1:
        # Parallelize
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net = net.cuda()

    net.train()
    criterion = torch.nn.BCEWithLogitsLoss().cuda()  # Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time': Timer(), 'val time': Timer()}
    validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


# Define the training function that takes in the data loader, the model, the loss function, the optimizer, and the epoch
def train(train_loader, net, criterion, optimizer, epoch):
    # Iterate over the batches in the data loader
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels for the current batch
        inputs, labels = data
        # Convert the inputs and labels to CUDA tensors
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        # Clear the gradients of all optimized tensors before computing the forward and backward pass
        optimizer.zero_grad()

        # Compute the output of the model for the current batch
        outputs = net(inputs)

        # Compute the loss between the predicted outputs and the ground truth labels
        loss = criterion(outputs, labels.unsqueeze(1).float())

        # Compute the gradients of the loss with respect to the model parameters
        loss.backward()

        # Update the model parameters based on the computed gradients and the optimization algorithm
        optimizer.step()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        # for binary classification
        outputs[outputs>0.5] = 1
        outputs[outputs<=0.5] = 0
        # for multi-classification ???

        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
    mean_iu = iou_/len(val_loader)   

    print('[mean iu %.4f]' % (mean_iu)) 
    net.train()
    criterion.cuda()


if __name__ == '__main__':
    main()








