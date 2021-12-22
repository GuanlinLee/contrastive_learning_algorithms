import time
import torch
import torch.utils.data
import models.linear
import models.densenet
import models.resnet
import models.mobilenetv2
import models.resnext
import models.vgg
import models.wrn
import models.encoder
import random
import numpy as np
from torch import nn
import utils
import os
import argparse
import DataLoader
from logging import getLogger
logger = getLogger()

parser = argparse.ArgumentParser(description='Contrastive Learning Training')
parser.add_argument( '--method', default='moco',
					 help='contrastive learning method')
parser.add_argument('--data_path', metavar='DIR',
					help='path to dataset to be sampled')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
					help='model architecture')
parser.add_argument('--class_num', default=10, type=int,
					help='class numbers (default: 200)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
					help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=0, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
					help='GPU id to use.')
parser.add_argument("--save", default='', type=str,
					help="name of checkpoint")
parser.add_argument("--save_path", default='', type=str,
					help="name of checkpoint for pre-trained model")
args = parser.parse_args()
def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
DIR = 'checkpoint_transfer/'+args.method+'/'+ args.arch + '/'+ args.dataset
if not os.path.exists(DIR):
	os.makedirs(DIR)
args.save = str(args.seed)
args.save_path = './checkpoint/'+args.method+'_'+args.dataset+'_'+args.arch+'.pkl'
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
torch.backends.cudnn.benchmark = True

def refer_basemodel(args):
	if 'dense' in args.arch:
		return getattr(models.densenet, args.arch)
	elif 'mobile' in args.arch:
		return getattr(models.mobilenetv2, args.arch)
	elif 'resnet' in args.arch:
		return getattr(models.resnet, args.arch)
	elif 'resnext' in args.arch:
		return getattr(models.resnext, args.arch)
	elif 'vgg' in args.arch:
		return getattr(models.vgg, args.arch)
	elif 'wrn' in args.arch:
		return getattr(models.wrn, args.arch)
	else:
		ValueError('base model no define')

def main():
	setup_seed(args.seed)
	logger = utils.create_logger(
		os.path.join(DIR, args.save + ".log"), rank=0
	)
	logger.info("============ Initialized logger ============")
	logger.info(
		"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
	)
	logger.info("The experiment will be stored in %s\n" % args.save+ ".pkl")
	logger.info("")
	best = 0.0
	if args.method == 'moco':
		model = models.encoder.MoCo(refer_basemodel(args), dim=args.dim, K=args.moco_k, m=args.moco_m, T=args.temperature, batch_size=args.batch_size)
	elif args.method == 'simclr':
		model = models.encoder.SimCLR(refer_basemodel(args), dim=args.dim)
	elif args.method == 'barlowtwins':
		model = models.encoder.Barlow_Twins(refer_basemodel(args), dim=args.dim)
	elif args.method == 'swav':
		model = models.encoder.SwAV(refer_basemodel(args), dim=args.dim, nmb_prototypes=args.nmb_prototypes)
	else:
		ValueError('method no define')
	model.cuda()
	checkpoint = torch.load(args.save_path)
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()
	dim = model(torch.randn(1,3,32,32).cuda(), True).size(1)
	Linear_model = models.linear.Linear(dim, args.class_num).cuda()
	criterion = nn.CrossEntropyLoss().cuda()

	optimizer = torch.optim.SGD(Linear_model.parameters(), args.lr,
									momentum=args.momentum,
									weight_decay=args.weight_decay)

	if 'stl' in args.dataset_transfer:
		resolution = 96
	elif 'tiny' in args.dataset_transfer:
		resolution = 64
	else:
		resolution = 32
	train_trans, test_trans = utils.make_aug(resolution)
	train_loader, test_loader = DataLoader.DL(args.dataset, args.data_path,
											  args.batch_size, train_trans,
											  test_trans, args.workers)
	for epoch in range(args.epochs):
		utils.adjust_learning_rate(optimizer, epoch, args)

		train(train_loader, model, Linear_model, criterion, optimizer, epoch, args)
		acc1 = evaluate(test_loader, model, Linear_model, criterion, epoch)
		if acc1 > best:
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': Linear_model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			}, is_best=True, filename=DIR+'/'+args.save+'.pkl', best_name=DIR+'/'+args.save+'_best.pkl')
			best = acc1
		else:
			utils.save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': Linear_model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			}, is_best=False, filename=DIR+'/'+args.save+'.pkl')
		logger.info('Best Accuracy on test set:'+ str(best))
def train(train_loader, model_base, model_linear, criterion, optimizer, epoch, args):
	batch_time = utils.AverageMeter('Time', ':6.3f')
	data_time = utils.AverageMeter('Data', ':6.3f')
	losses = utils.AverageMeter('Loss', ':.4e')
	top1 = utils.AverageMeter('Top1_ACC', ':.4e')
	top5= utils.AverageMeter('Top5_ACC', ':.4e')
	model_linear.train()

	end = time.time()
	for i, (images, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		images = images.cuda()
		target = target.cuda()
		# compute output
		feat = model_base(images, True)
		output = model_linear(feat)
		loss = criterion(output, target)
		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
		# measure accuracy and record loss
		losses.update(loss.item(), images.size(0))
		top1.update(prec1.item(), images.size(0))
		top5.update(prec5.item(), images.size(0))
		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			logger.info(
				"Epoch: [{0}][{1}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Top1 ACC {top1.val:.4f} ({top1.avg:.4f})\t"
				"Top5 ACC {top5.val:.4f} ({top5.avg:.4f})\t"
				"Lr: {lr:.4f}".format(
					epoch,
					i,
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					top1=top1,
					top5=top5,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

def evaluate(test_loader, model_base, model_linear, criterion, epoch):
	batch_time = utils.AverageMeter('Time', ':6.3f')
	data_time = utils.AverageMeter('Data', ':6.3f')
	losses = utils.AverageMeter('Loss', ':.4e')
	top1 = utils.AverageMeter('Top1_ACC', ':.4e')
	top5= utils.AverageMeter('Top5_ACC', ':.4e')
	model_linear.eval()

	end = time.time()
	for i, (images, target) in enumerate(test_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		images = images.cuda()
		target = target.cuda()
		# compute output
		feat = model_base(images, True)
		output = model_linear(feat)
		loss = criterion(output, target)
		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		# measure accuracy and record loss
		prec1, prec5 = utils.accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.item(), images.size(0))
		top1.update(prec1.item(), images.size(0))
		top5.update(prec5.item(), images.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()


	logger.info(
		"Epoch: [{0}]\t"
		"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
		"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
		"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
		"Top1 ACC {top1.val:.4f} ({top1.avg:.4f})\t"
		"Top5 ACC {top5.val:.4f} ({top5.avg:.4f})\t"
			.format(
			epoch,
			batch_time=batch_time,
			data_time=data_time,
			loss=losses,
			top1=top1,
			top5=top5,
		)
	)
	return top1.avg

if __name__ == '__main__':
	main()