import torch.nn as nn

class Linear(nn.Module):
	def __init__(self, dim, class_num):
		super(Linear, self).__init__()
		self.linear = nn.Linear(dim, class_num)
	def forward(self, x):
		return self.linear(x)