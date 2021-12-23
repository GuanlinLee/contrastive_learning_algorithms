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
parser.add_argument('--seed', default=0, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
					help='GPU id to use.')
parser.add_argument("--save", default='', type=str,
					help="name of checkpoint")
parser.add_argument("--save_path", default='', type=str,
					help="name of checkpoint for pre-trained model")
parser.add_argument("--temperature", default=0.1, type=float,
					help="temperature parameter in training loss")
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
	dim = model(torch.randn((1,3,32,32)).cuda(), feature_only = True).size(1)
	Linear_model = models.linear.Linear(dim, args.class_num).cuda()
	checkpoint = torch.load(DIR+'/'+args.save+'_best.pkl')
	Linear_model.load_state_dict(checkpoint['state_dict'])
	Linear_model.eval()
	criterion = nn.CrossEntropyLoss().cuda()

	if 'stl' in args.dataset_transfer:
		resolution = 96
	elif 'tiny' in args.dataset_transfer:
		resolution = 64
	else:
		resolution = 32
	train_trans, test_trans = utils.make_aug_transfer(resolution, args.dataset)
	train_loader, test_loader = DataLoader.DL(args.dataset, args.data_path,
											  args.batch_size, train_trans,
											  test_trans, args.workers)

	acc1 = evaluate(test_loader, model, Linear_model, criterion)
	print(acc1)

def evaluate(test_loader, model_base, model_linear, criterion):
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
		feat = model_base(images, feature_only = True)
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
	return top1.avg

if __name__ == '__main__':
	main()