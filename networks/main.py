import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import model_list
import sys
import gc
cwd = os.getcwd()
sys.path.append(cwd+'/../')
import datasets as datasets
#import datasets.transforms as transforms
from torchvision import transforms

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ocrnet',
                    help='model architecture (default: ocrnet)')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.90, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', #../checkpoint_v1_1.pth.tar
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=False, help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')

best_prec1 = 0

torch.manual_seed(1)
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(1)
    device = torch.device("cuda")

def main():
    global args, best_prec1
    args = parser.parse_args()

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    character_set = "-0123456789#"  # - is the blank symbol for ctc loss
    # character_set = ['-', '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    #      '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    #      '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
    #      '新',
    #      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    #      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    #      'W', 'X', 'Y', 'Z', 'I', '#'
    #      ]
    #
    # character_set = "".join(character_set)

    # create model
    if args.arch == 'ocrnet':
        model = model_list.ocrnet(device=device, num_classes=len(character_set))
    else:
        raise Exception('Model not supported yet')

    if not args.distributed:
        if args.arch.startswith('ocrnet') or args.arch.startswith('vgg'):
            #model.backbone = torch.nn.DataParallel(model.backbone)
            pass
        else:
            model = torch.nn.DataParallel(model).to(device)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    #criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.CTCLoss(blank=character_set.index('-'), reduction='mean',zero_infinity=False)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr,
    #                             #momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9, eps=1e-08,
                              momentum=0.9, weight_decay=2e-5)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        if args.pretrained:
            model_path = 'sirius_ai_checkpoint.pth'
            if torch.cuda.is_available():
                pretrained_model = torch.load(model_path)
            else:
                pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_model)
            print("Loaded pretrained model.")
        # else:
        #     model.init_weights()
    # won't harm anyway but if checkpoint loaded its needed
    model.to(device)

    cudnn.benchmark = True

    train_dataset = datasets.MyDataset(
        img_dir='../train_images_48x144/',
        transform=transforms.Compose([
            #transforms.ToTensor(),
            #transforms.Resize([92,24])
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), character_set=character_set)

    validation_dataset = datasets.MyDataset(
        img_dir='../test_images_48x144/',
        transform=transforms.Compose([
            #transforms.ToTensor(),
            #transforms.Resize([92,24])
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]), character_set=character_set)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    print(model)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, emb_len, target_len) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #print(target.shape)
        #target = target.long()
        input = input.to(device)
        target = target.to(device)
        target_len = target_len.to(device)
        emb_len = emb_len.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_len = torch.autograd.Variable(target_len)
        emb_len = torch.autograd.Variable(emb_len)

        # compute output
        output = model(input_var)
        #output = output.permute(0,2,1)
        #output = output.permute(1, 0, 2) #for ctc loss
        output = torch.nn.functional.log_softmax(output, 2)
        loss = criterion(output, target_var, emb_len, target_len)

        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, emb_len, target_len) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        target_len = target_len.to(device)
        emb_len = emb_len.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_len = torch.autograd.Variable(target_len)
        emb_len = torch.autograd.Variable(emb_len)

        # compute output
        output = model(input_var)
        #output = output.permute(1, 0, 2)
        output = torch.nn.functional.log_softmax(output, 2)
        loss = criterion(output, target_var, emb_len, target_len)

        losses.update(loss.data, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 2 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 40))
#     print('Learning rate:', lr)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr

def adjust_learning_rate(optimizer, cur_epoch, base_lr=0.1, lr_schedule=[4, 8, 12, 14, 16]):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
