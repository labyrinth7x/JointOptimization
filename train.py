import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import logging
import os
import time
from dataset.cifar10 import get_dataset
#from utils.visualize import save_fig
from utils.criterion import accuracy_v2, joint_opt_loss
from utils.AverageMeter import AverageMeter
import torch.utils.data as data
from torch import optim
from models.resnet import resnet18,resnet34


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1,help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset_type', default='sym_noise', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./data/img_data/train', help='root for train data')
    parser.add_argument('--test_root', default='./data/img_data/test', help='root for test data')
    parser.add_argument('--epoch_begin', default=70, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=10, help='#epoch to average to update soft labels')
    parser.add_argument('--noise_ratio', type=float, default=0.1, help='percent of noise')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyper param for loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyper param for loss')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='resnet34', help='the backbone of the network')
    args = parser.parse_args()
    return args


def data_config(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train, val, test = get_dataset(args, transform_train, transform_val)
    train_loader = data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print('-------> Data loading')
    return train_loader, val_loader


def record_params(args):
    dst_folder = args.out + '/' + args.dataset_type + '-lr-{}-ratio-{}-alpha-{}-beta-{}-{}'.format(args.lr, args.noise_ratio, args.alpha, args.beta, args.network)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    rd = open(dst_folder + '/config.txt', 'w')
    rd.write('lr:%f'%args.lr + '\n')
    rd.write('wd:%f'%args.wd + '\n')
    rd.write('momentum:%f'%args.momentum + '\n')
    rd.write('batch_size:%d'%args.batch_size + '\n')
    rd.write('epoch:%d'%args.epoch + '\n')
    rd.write('dataset_type:'+args.dataset_type)
    rd.write('noise_ratio:%f'%args.noise_ratio +'\n')
    rd.close()

    handler = logging.FileHandler(dst_folder + '/train.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    dst_folder = dst_folder + '/first_train'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    return dst_folder

def record_result(dst, best_ac):
    rd = open(dst, 'a+')
    rd.write('best_ac:%.3f'%best_ac)
    rd.close()


def network_config(args):
    if args.network == 'renset34':
        network = resnet34()
        print('-------> Creating network resnet-34')
    else:
        network = resnet18()
        print('-------> Creating network resnet-18')
    network = torch.nn.DataParallel(network).cuda()
    print('Total params: %2.fM' % (sum(p.numel() for p in network.parameters()) / 1000000.0))
    cudnn.benchmark = True
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    return network, optimizer, use_cuda


def save_checkpoint(state, dst_folder, epoch):
    dst = dst_folder + '/epoch-' + str(epoch) + '.pkl'
    torch.save(state, dst)


def train(train_loader, network, optimizer, use_cuda, args):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    network.train()

    end = time.time()

    results = np.zeros((len(train_loader.dataset), 10), dtype=np.float32)
    for images, labels, soft_labels, index in train_loader:
        if use_cuda:
            images = images.cuda()
            labels = labels.cuda(non_blocking=True)
            soft_labels = soft_labels.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
        
        # compute output
        outputs = network(images)
        prob, loss = joint_opt_loss(outputs, soft_labels, use_cuda, args)
        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1,5])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # update soft labels
    train_loader.dataset.update_labels(results)
    return train_loss.avg, top5.avg, top1.avg, batch_time.sum


def validate(val_loader, network, criterion, use_cuda):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_loss = AverageMeter()

    # switch to evaluate mode
    network.eval()

    with torch.no_grad():
        end = time.time()
        for images, labels in val_loader:
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda(non_blocking=True)
            outputs = network(images)
            prec1, prec5 = accuracy_v2(outputs, labels, top=[1,5])
            loss = criterion(outputs, labels)

            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))
            val_loss.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    return top5.avg, top1.avg, batch_time.sum


def main(args, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0

    # data lodaer
    train_loader, val_loader = data_config(args)

    # criterion
    val_criterion = nn.CrossEntropyLoss()

    # network config
    network, optimizer, use_cuda = network_config(args)

    for epoch in range(args.epoch):
        # train for one epoch
        train_loss, top_5_train_ac, top1_train_ac, train_time = train(train_loader, network, optimizer, use_cuda, args)
        # evaluate on validation set
        top5_val_ac, top1_val_ac, val_time = validate(val_loader, network, val_criterion, use_cuda)
        # remember best prec@1, save checkpoint and logging to the console.
        if top1_val_ac >= best_ac:
            state = {'state_dict': network.state_dict(), 'epoch': epoch, 'ac': [top5_val_ac, top1_val_ac], 'best_ac': best_ac, 'time': [train_time, val_time]}
            best_ac = top1_val_ac
            # save model
            save_checkpoint(state, dst_folder, epoch)
            # logging
        logging.info('\nEpoch: [{}|{}], train_loss: {:.3f}, top1_train_ac: {:.3f}, top5_val_ac: {:.3f}, top1_val_ac: {:.3f}, val_time: {:.3f}, train_time: {:.3f}'.format(epoch, args.epoch, train_loss, top1_train_ac, top5_val_ac, top1_val_ac, val_time, train_time))

    #save_fig(dst_folder)
    print('Best ac:%f'%best_ac)
    record_result(dst_folder + '/config.txt', best_ac)


if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    # record params
    dst_folder = record_params(args)
    # train
    main(args, dst_folder)
