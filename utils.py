import torchvision.transforms as transforms
from PIL import ImageFilter
import random
import math
import torch
import shutil
import time
from datetime import timedelta
import logging
class TwoCropsTransform:
	"""Take two random crops of one image as the query and key."""

	def __init__(self, base_transform):
		self.base_transform = base_transform

	def __call__(self, x):
		q = self.base_transform(x)
		k = self.base_transform(x)
		return [q, k]

class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x

def make_aug(resolution):
	train_augmentation = [
		transforms.RandomResizedCrop(resolution, scale=(0.2, 1.)),
		transforms.RandomApply([
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
		], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
										 std=[0.229, 0.224, 0.225])
	]
	test_augmentation = [
		transforms.Resize(resolution),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406],
							 std=[0.229, 0.224, 0.225])
	]
	return TwoCropsTransform(transforms.Compose(train_augmentation)), transforms.Compose(test_augmentation)

def make_aug_transfer(resolution, dataset):
	if 'cifar' in dataset:
		norm = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
										   std=[0.247, 0.243, 0.261])
	else:
		norm = transforms.Normalize(mean=[0.4376821, 0.4437697, 0.47280442],
									std=[0.19803012, 0.20101562, 0.19703614])
	train_augmentation = [
		transforms.RandomResizedCrop(resolution, scale=(0.2, 1.)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		norm
	]
	test_augmentation = [
		transforms.Resize(resolution),
		transforms.ToTensor(),
		norm
	]
	return transforms.Compose(train_augmentation), transforms.Compose(test_augmentation)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, best_name)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
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

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		print('\t'.join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = '{:' + str(num_digits) + 'd}'
		return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
	"""Decay the learning rate based on schedule"""
	lr = args.lr
	lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
	"""Computes the accuracy over the k top predictions for the specified values of k"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

class LogFormatter:
	def __init__(self):
		self.start_time = time.time()

	def format(self, record):
		elapsed_seconds = round(record.created - self.start_time)

		prefix = "%s - %s - %s" % (
			record.levelname,
			time.strftime("%x %X"),
			timedelta(seconds=elapsed_seconds),
		)
		message = record.getMessage()
		message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
		return "%s - %s" % (prefix, message) if message else ""

def create_logger(filepath, rank):
	"""
	Create a logger.
	Use a different log file for each process.
	"""
	# create log formatter
	log_formatter = LogFormatter()

	# create file handler and set level to debug
	if filepath is not None:
		if rank > 0:
			filepath = "%s-%i" % (filepath, rank)
		file_handler = logging.FileHandler(filepath, "a")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(log_formatter)

	# create console handler and set level to info
	console_handler = logging.StreamHandler()
	console_handler.setLevel(logging.INFO)
	console_handler.setFormatter(log_formatter)

	# create logger and set level to debug
	logger = logging.getLogger()
	logger.handlers = []
	logger.setLevel(logging.DEBUG)
	logger.propagate = False
	if filepath is not None:
		logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# reset logger elapsed time
	def reset_time():
		log_formatter.start_time = time.time()

	logger.reset_time = reset_time

	return logger