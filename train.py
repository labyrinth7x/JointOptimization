import torch
import torchvision.transforms as transforms
import argparse
import logging
import os
from dataset.cifar10 import get_dataset
from utils.visualize import save_fig
from utils.criterion import accuracy
import torch.utils.data as data
from torch import optim
from models.resnet import resnet18

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')

    parser.add_argument('--lr', type=float, default=0.1, )
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--dataset_type', default='sym_noise', help='noise type of the dataset')
    parser.add_argument('--train_root', help='root for train data')
    parser.add_argument('--test_root', help='root for test data')
    parser.add_argument('--epoch_begin', default=70, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', default=10, help='#epoch to average to update soft labels')
    parser.add_argument('--noise_ratio', type=float, default=0.0, help='percent of symmetric noise')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', type=str, help='Directory of the output')
    parser.add_argument('--alpha', type=float, default=1.0, help='Hyper param for loss')
    parser.add_argument('--beta', type=float, default=0.5, help='Hyper param for loss')

    args = parser.parse_args()

def data_config(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(),
    ])
    train, val = get_dataset(args, transform_train=transform_train, transform_val=transform_val)
    train_loader = data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

def record(args):
    dst_folder = args.out + '/' + args.dataset_type + '-lr-{}-ratio-{}'.format(args.lr, args.noise_ratio)
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    rd = open(dst_folder + 'config.txt', 'w')
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

    return dst_folder

def network_config(args):
    network = resnet18()
    optimizer = optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return network, optimizer, device


def save_checkpoint(state, dst_folder, epoch):

    dst = dst_folder + '/epoch-' + str(epoch) + '.pkl'
    torch.save(state, dst)


def train(train_loader, network, optimizer, device):


def validate(val_loader, network, optimizer, device):
    # test the model
    top1 = 0
    top5 = 0
    network.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            top5_num, top1_num = accuracy(outputs, labels, top=[1,5])
            top1 += top1_num
            top5 += top5_num
    return float(top5) / val_loader.__len__(), float(top1) / val_loader.__len__()


def main(args, dst_folder):
    # best_ac only record the best top1_ac for validation set.
    best_ac = 0.0
    if not os.path.isdir(args.out):
        os.makedirs(args.out)

    # data lodaer
    train_loader, val_loader = data_config(args)

    # network config
    network, optimizer, device = network_config(args)

    for epoch in range(args.epoch):
        train_loss, train_ac = train(train_loader, network, optimizer, device)
        top5_val_ac, top1_val_ac = validate(val_loader, network, optimizer, device)

        if top1_val_ac >= best_ac:
            state = {'state_dict':network.state_dict(), 'epoch':epoch, 'ac': [top5_val_ac, top1_val_ac], 'best_ac':best_ac}
            best_ac = top1_val_ac
            # save model
            save_checkpoint(state, dst_folder, epoch)
            # logging
            logging.info('\nEpoch: [{}|{}], train_loss: {}, train_ac: {}, top5_val_ac: {}, top1_val_ac: {}'.format(epoch, args.epoch, train_loss, train_ac, top5_val_ac, top1_val_ac))

    save_fig(dst_folder)
    print('Best ac:%f'%best_ac)




if __name__=="__main__":
    args = parse_args()
    logging.info(args)
    # record
    dst_folder = record(args)
    # train
    main(args, dst_folder)
