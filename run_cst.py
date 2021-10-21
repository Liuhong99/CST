import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
from randaugment import rand_augment_transform, GaussianBlur
from fix_utils import ImageClassifier
from sam import SAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)

def entropy(predictions: torch.Tensor, reduction='none') -> torch.Tensor:
  
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H

class TsallisEntropy(nn.Module):
    
    def __init__(self, temperature: float, alpha: float):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        N, C = logits.shape
        
        pred = F.softmax(logits / self.temperature, dim=1) 
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  
        
        sum_dim = torch.sum(pred * entropy_weight, dim = 0).unsqueeze(dim=0)
      
        return 1 / (self.alpha - 1) * torch.sum((1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim = -1)))

class TransformFixMatch(object):
    def __init__(self):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.weak = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            T.ToTensor(),
            normalize,
        ])
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong
    

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    

    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])

    unlabeled_transform = TransformFixMatch()

    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=unlabeled_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes
    args.num_cls = num_classes
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    # define optimizer and lr scheduler

    base_optimizer = SGD
    optimizer = SAM(classifier.get_parameters(), base_optimizer, lr = args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, adaptive = True, rho = args.rho)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    ts_loss = TsallisEntropy(temperature=args.temperature, alpha = args.alpha)

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    if args.phase == 'analysis':
        # extract features from both domains
        feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
        source_feature = collect_feature(train_source_loader, feature_extractor, device)
        target_feature = collect_feature(train_target_loader, feature_extractor, device)
        # plot t-SNE
        tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
        tsne.visualize(source_feature, target_feature, tSNE_filename)
        print("Saving t-SNE to", tSNE_filename)
        # calculate A-distance, which is a measure for distribution discrepancy
        A_distance = a_distance.calculate(source_feature, target_feature, device)
        print("A-distance =", A_distance)
        return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(min(args.epochs, args.early)):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(train_source_iter, train_target_iter, classifier, ts_loss, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, ts: TsallisEntropy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    trans_losses = AverageMeter('Trans Loss', ':3.2f')
    rev_losses = AverageMeter('CST Loss', ':3.2f')
    fix_losses = AverageMeter('Fix Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, trans_losses, rev_losses, fix_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i in range(args.iters_per_epoch):
        x_s, labels_s = next(train_source_iter)
        (x_t, x_t_u), _ = next(train_target_iter)

        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_s, x_t), dim=0)
        y, f = model(x)
        y_t_u, _ = model(x_t_u)

        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u,
                              reduction='none') * max_prob.ge(args.threshold).float().detach()).mean()

        # compute cst
        target_data_train_r = f_t
        target_data_train_r = target_data_train_r / (torch.norm(target_data_train_r, dim = -1).reshape(target_data_train_r.shape[0], 1))
        target_data_test_r = f_s
        target_data_test_r = target_data_test_r / (torch.norm(target_data_test_r, dim = -1).reshape(target_data_test_r.shape[0], 1))
        target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
        target_kernel_r = target_gram_r
        test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
        test_kernel_r = test_gram_r
        target_train_label_r = torch.nn.functional.one_hot(pred_u, args.num_cls) - 1 / float(args.num_cls) 
        target_test_pred_r = test_kernel_r.mm(torch.inverse(target_kernel_r + 0.001 * torch.eye(args.batch_size).cuda())).mm(target_train_label_r)
        reverse_loss = nn.MSELoss()(target_test_pred_r, torch.nn.functional.one_hot(labels_s, args.num_cls) - 1 / float(args.num_cls)) 

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)

        if Lu != 0:
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else: 
            loss = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))
        trans_losses.update(transfer_loss.item(), x_s.size(0))
        rev_losses.update(reverse_loss.item(), x_s.size(0))
        fix_losses.update(Lu.item(), x_s.size(0))


        # compute gradient and do the first SGD step
        loss.backward()
        optimizer.first_step(zero_grad=True)
        lr_scheduler.step()

        # compute gradient and do the second SGD step

        y, f = model(x)
        y_t_u, _ = model(x_t_u)

        f_s, f_t = f.chunk(2, dim=0)
        y_s, y_t = y.chunk(2, dim=0)

        # generate target pseudo-labels
        max_prob, pred_u = torch.max(F.softmax(y_t), dim=-1)
        Lu = (F.cross_entropy(y_t_u, pred_u,
                              reduction='none') * max_prob.ge(args.threshold).float().detach()).mean()

        # compute cst
        target_data_train_r = f_t
        target_data_train_r = target_data_train_r / (torch.norm(target_data_train_r, dim = -1).reshape(target_data_train_r.shape[0], 1))
        target_data_test_r = f_s
        target_data_test_r = target_data_test_r / (torch.norm(target_data_test_r, dim = -1).reshape(target_data_test_r.shape[0], 1))
        target_gram_r = torch.clamp(target_data_train_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
        target_kernel_r = target_gram_r
        test_gram_r = torch.clamp(target_data_test_r.mm(target_data_train_r.transpose(dim0 = 1, dim1 = 0)),-0.99999999,0.99999999)
        test_kernel_r = test_gram_r
        target_train_label_r = torch.nn.functional.one_hot(pred_u, args.num_cls) - 1 / float(args.num_cls) 
        target_test_pred_r = test_kernel_r.mm(torch.inverse(target_kernel_r + 0.001 * torch.eye(args.batch_size).cuda())).mm(target_train_label_r)
        reverse_loss = nn.MSELoss()(target_test_pred_r, torch.nn.functional.one_hot(labels_s, args.num_cls) - 1 / float(args.num_cls)) 

        cls_loss = F.cross_entropy(y_s, labels_s)
        transfer_loss = ts(y_t)

        if Lu != 0:
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1 + Lu * args.trade_off3
        else: 
            loss1 = cls_loss + transfer_loss * args.trade_off + reverse_loss * args.trade_off1

        loss1.backward()
        optimizer.second_step(zero_grad=True)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='CST for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling')
    parser.add_argument('--alpha', default= 1.9, type=float,
                        help='the entropic index of Tsallis loss')
    parser.add_argument('--trade-off', default= 0.08, type=float,
                        help='the trade-off hyper-parameter for  Tsallis entropy')
    parser.add_argument('--trade-off1', default=0.5, type=float,
                        help='the trade-off hyper-parameter for CST loss')
    parser.add_argument('--trade-off3', default=0.5, type=float,
                        help='the trade-off hyper-parameter for fix loss')
    parser.add_argument('--threshold', default=0.97, type=float)
    parser.add_argument('--rho', default=0.5, type=float,
                     help='rho',
                    dest='rho')

    # training parameters
    parser.add_argument('-b', '--batch-size', default=28, type=int,
                        metavar='N',
                        help='mini-batch size (default: 28)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early', default=20, type=int, metavar='N',
                        help='number of total epochs to early stopping')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='cst',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
