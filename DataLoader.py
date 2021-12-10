import torch
import torchvision
import torchvision.datasets
def DL(dataset, path, batch_size, train_trans, test_trans, worker):
		if dataset == 'cifar10':
			train_data = torchvision.datasets.CIFAR10(root=path, train=True, transform=train_trans, download=True)
			test_data = torchvision.datasets.CIFAR10(root=path, train=False, transform=test_trans, download=True)
		elif dataset == 'cifar100':
			train_data = torchvision.datasets.CIFAR100(root=path, train=True, transform=train_trans, download=True)
			test_data = torchvision.datasets.CIFAR100(root=path, train=False, transform=test_trans, download=True)
		elif dataset == 'stl10':
			train_data = torchvision.datasets.STL10(root=path, split="unlabeled", transform=train_trans, download=True)
			test_data = torchvision.datasets.STL10(root=path, split="test", transform=test_trans, download=True)
		elif dataset == 'tiny_imagenet':
			train_data = torchvision.datasets.ImageFolder(path+'/train', train_trans)
			test_data = torchvision.datasets.ImageFolder(path+'/val', test_trans)
		else:
			ValueError('dataset no define')
		'''
		train_data = torchvision.datasets.ImageFolder(path, transform=train_trans)
		train_queue = torch.utils.data.DataLoader(
			train_data, batch_size=batch_size,
			pin_memory=True, num_workers=worker)
		'''
		train_loader = torch.utils.data.DataLoader(
			train_data, batch_size=batch_size, shuffle=True,
			num_workers=worker, pin_memory=True, drop_last=True)
		test_loader = torch.utils.data.DataLoader(
			test_data, batch_size=batch_size, shuffle=False,
			num_workers=worker, pin_memory=True, drop_last=False)

		return train_loader, test_loader
