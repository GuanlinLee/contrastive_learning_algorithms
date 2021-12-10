import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import utils
from utils import *
import DataLoader
import models.encoder
import models.densenet
import models.resnet
import models.mobilenetv2
import models.resnext
import models.vgg
import models.wrn

from logging import getLogger
logger = getLogger()

parser = argparse.ArgumentParser(description='Contrastive Learning Training')
parser.add_argument( '--method', default='moco',
					help='contrastive learning method')
parser.add_argument('--data_path', metavar='DIR',
					help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
					help='which dataset used to train')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
					help='model architecture')
parser.add_argument('--dim', default=128, type=int,
					help='feature dimension (default: 128)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
					help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
					metavar='N',
					help='mini-batch size (default: 256), this is the total '
						 'batch size of all GPUs on the current node when '
						 'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
					metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)',
					dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument("--temperature", default=0.1, type=float,
					help="temperature parameter in training loss")
parser.add_argument("--save", default='', type=str,
					help="name of checkpoint")

# moco specific configs:

parser.add_argument('--moco_k', default=65536, type=int,
					help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.999, type=float,
					help='moco momentum of updating key encoder (default: 0.999)')


# barlow twins specific configs:
parser.add_argument('--lmbda', default=0.0078125, type=float, help='Lambda that controls the on- and off-diagonal terms')


# swav specific configs:
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
					help="list of number of crops (example: [2, 6])")
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
					help="list of crops id used for computing assignments")
parser.add_argument("--epsilon", default=0.05, type=float,
					help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
					help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
					help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
					help="length of the queue (0 for no queue)")
parser.add_argument("--epoch_queue_starts", type=int, default=15,
					help="from this epoch, we start using a queue")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
					help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
					help="initial warmup learning rate")
args = parser.parse_args()
if not os.path.exists('checkpoint'):
	os.mkdir('checkpoint')
args.save = args.method+'_'+args.dataset+'_'+args.arch
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
	logger = create_logger(
		os.path.join('checkpoint', args.save + ".log"), rank=0
	)
	logger.info("============ Initialized logger ============")
	logger.info(
		"\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
	)
	logger.info("The experiment will be stored in %s\n" % args.save+ ".pkl")
	logger.info("")
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
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], args.lr,
							momentum=args.momentum,
							weight_decay=args.weight_decay)
	if 'cifar' in args.dataset:
		resolution = 32
	elif 'stl' in args.dataset:
		resolution = 96
	elif 'tiny' in args.dataset:
		resolution = 64
	train_trans, test_trans = utils.make_aug(resolution)
	train_loader, _ = DataLoader.DL(args.dataset, args.data_path, args.batch_size, train_trans, test_trans, args.workers)
	for epoch in range(args.epochs):
		adjust_learning_rate(optimizer, epoch, args)
		if args.method == 'moco':
			train_moco(train_loader, model, criterion, optimizer, epoch, args)
		elif args.method == 'simclr':
			train_simclr(train_loader, model, optimizer, epoch, args)
		elif args.method == 'barlowtwins':
			train_barlowtwins(train_loader, model, optimizer, epoch, args)
		elif args.method == 'swav':
			train_swav(train_loader, model, optimizer, epoch, args)
		save_checkpoint({
				'epoch': epoch + 1,
				'arch': args.arch,
				'state_dict': model.state_dict(),
				'optimizer' : optimizer.state_dict(),
			}, is_best=False, filename='checkpoint/'+args.save+'.pkl')

def train_moco(train_loader, model, criterion, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')

	# switch to train mode
	model.train()

	end = time.time()
	for i, (images, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		images[0] = images[0].cuda(non_blocking=True)
		images[1] = images[1].cuda(non_blocking=True)

		# compute output
		output, target = model(images[0], images[1])
		loss = criterion(output, target)

		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		# measure accuracy and record loss
		losses.update(loss.item(), images[0].size(0))

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
				"Lr: {lr:.4f}".format(
					epoch,
					i,
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

def train_swav(train_loader, model, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	softmax = nn.Softmax(dim=1).cuda()
	model.train()

	end = time.time()
	for i, (images, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		images[0] = images[0].cuda(non_blocking=True)
		images[1] = images[1].cuda(non_blocking=True)
		iteration = epoch * len(train_loader) + i
		# normalize the prototypes
		with torch.no_grad():
			w = model.prototypes.weight.data.clone()
			w = F.normalize(w, dim=1, p=2)
			model.prototypes.weight.copy_(w)

		# ============ multi-res forward passes ... ============
		embedding, output = model(images)
		bs = images[0].size(0)

		# ============ swav loss ... ============
		loss = 0
		for _, crop_id in enumerate(args.crops_for_assign):
			with torch.no_grad():
				out = output[bs * crop_id: bs * (crop_id + 1)].detach()
				# get assignments
				q = torch.exp(out / args.epsilon).t()
				q = distributed_sinkhorn(q, args.sinkhorn_iterations)[-bs:]

			# cluster assignment prediction
			subloss = 0
			for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
				p = softmax(output[bs * v: bs * (v + 1)] / args.temperature)
				subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
			loss += subloss / (np.sum(args.nmb_crops) - 1)
		loss /= len(args.crops_for_assign)

		# ============ backward and optim step ... ============
		optimizer.zero_grad()
		loss.backward()
		# cancel gradients for the prototypes
		if iteration < args.freeze_prototypes_niters:
			for name, p in model.named_parameters():
				if "prototypes" in name:
					p.grad = None
		optimizer.step()

		# ============ misc ... ============
		losses.update(loss.item(), images[0].size(0))
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			logger.info(
				"Epoch: [{0}][{1}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Lr: {lr:.4f}".format(
					epoch,
					i,
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

def distributed_sinkhorn(Q, nmb_iters):
	with torch.no_grad():
		sum_Q = torch.sum(Q)
		# dist.all_reduce(sum_Q)
		Q /= sum_Q

		u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
		r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
		c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (1.0 * Q.shape[1])

		curr_sum = torch.sum(Q, dim=1)
		# dist.all_reduce(curr_sum)

		for it in range(nmb_iters):
			u = curr_sum
			Q *= (r / u).unsqueeze(1)
			Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
			curr_sum = torch.sum(Q, dim=1)
			# dist.all_reduce(curr_sum)
		return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

def train_simclr(train_loader, model, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	model.train()

	end = time.time()
	for i, (images, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		images[0] = images[0].cuda(non_blocking=True)
		images[1] = images[1].cuda(non_blocking=True)
		batch_size = images[0].size(0)
		feature_1, out_1 = model(images[0])
		feature_2, out_2 = model(images[1])
		# [2*B, D]
		out = torch.cat([out_1, out_2], dim=0)
		# [2*B, 2*B]
		sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
		mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
		# [2*B, 2*B-1]
		sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

		# compute loss
		pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / args.temperature)
		# [2*B]
		pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
		loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.update(loss.item(), images[0].size(0))
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			logger.info(
				"Epoch: [{0}][{1}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Lr: {lr:.4f}".format(
					epoch,
					i,
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

def train_barlowtwins(train_loader, model, optimizer, epoch, args):
	batch_time = AverageMeter('Time', ':6.3f')
	data_time = AverageMeter('Data', ':6.3f')
	losses = AverageMeter('Loss', ':.4e')
	model.train()

	end = time.time()
	for i, (images, _) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		images[0] = images[0].cuda(non_blocking=True)
		images[1] = images[1].cuda(non_blocking=True)
		batch_size = images[0].size(0)
		feature_1, out_1 = model(images[0])
		feature_2, out_2 = model(images[1])
		# normalize the representations along the batch dimension
		out_1_norm = (out_1 - out_1.mean(dim=0)) / out_1.std(dim=0)
		out_2_norm = (out_2 - out_2.mean(dim=0)) / out_2.std(dim=0)

		# cross-correlation matrix
		c = torch.matmul(out_1_norm.T, out_2_norm) / batch_size

		# loss
		on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

		off_diag = off_diagonal(c).pow_(2).sum()

		loss = on_diag + args.lmbda * off_diag
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.update(loss.item(), images[0].size(0))
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			logger.info(
				"Epoch: [{0}][{1}]\t"
				"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
				"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
				"Loss {loss.val:.4f} ({loss.avg:.4f})\t"
				"Lr: {lr:.4f}".format(
					epoch,
					i,
					batch_time=batch_time,
					data_time=data_time,
					loss=losses,
					lr=optimizer.param_groups[0]["lr"],
				)
			)

def off_diagonal(x):
	# return a flattened view of the off-diagonal elements of a square matrix
	n, m = x.shape
	assert n == m
	return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

if __name__ == '__main__':
	main()