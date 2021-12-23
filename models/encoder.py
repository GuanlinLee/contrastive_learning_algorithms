import torch
import torch.nn as nn
import torch.nn.functional as F

class Dummy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x
#MoCo V2
class MoCo(nn.Module):
	"""
	Build a MoCo model with: a query encoder, a key encoder, and a queue
	https://arxiv.org/abs/1911.05722
	"""
	def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, batch_size=64):
		"""
		dim: feature dimension (default: 128)
		K: queue size; number of negative keys (default: 65536)
		m: moco momentum of updating key encoder (default: 0.999)
		T: softmax temperature (default: 0.07)
		"""
		super(MoCo, self).__init__()

		self.K = K
		self.m = m
		self.T = T
		self.batch_size = batch_size

		# create the encoders
		# num_classes is the output fc dimension
		self.encoder_q = base_encoder(num_classes=dim)
		self.encoder_k = base_encoder(num_classes=dim)



		self.encoder_q.fc = nn.Sequential(nn.Linear(self.encoder_q.fc.weight.shape[1], self.encoder_q.fc.weight.shape[1]),
										  nn.ReLU(), nn.Linear(self.encoder_q.fc.weight.shape[1], dim))

		self.encoder_k.fc = nn.Sequential(nn.Linear(self.encoder_k.fc.weight.shape[1], self.encoder_k.fc.weight.shape[1]),
										  nn.ReLU(), nn.Linear(self.encoder_k.fc.weight.shape[1], dim))

		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  # initialize
			param_k.requires_grad = False  # not update by gradient
		self.q_fc = self.encoder_q.fc
		self.k_fc = self.encoder_k.fc
		self.encoder_q.fc = Dummy()
		self.encoder_k.fc = Dummy()
		# create the queue
		self.register_buffer("queue",  nn.functional.normalize(torch.randn(self.K, dim, requires_grad=False), dim=1)/10)
		self.ptr = 0

	@torch.no_grad()
	def update_k_encoder_weights(self):
		""" manually update key encoder weights with momentum and no_grad"""
		# update k_encoder.parameters
		for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			p_k.data = p_k.data*self.m + (1.0 - self.m)*p_q.data
			p_k.requires_grad = False

		# update k_fc.parameters
		for p_q, p_k in zip(self.q_fc.parameters(), self.k_fc.parameters()):
			p_k.data = p_k.data*self.m + (1.0 - self.m)*p_q.data
			p_k.requires_grad = False

	@torch.no_grad()
	def update_queue(self, k):
		""" swap oldest batch with the current key batch and update ptr"""
		self.queue[self.ptr: self.ptr + self.batch_size, :] = k.detach().cpu()
		self.ptr = (self.ptr + self.batch_size) % self.K
		self.queue.requires_grad = False

	def forward(self, *args, feature_only=False, prints=False):
		if feature_only:
			return self.feature_forward(*args)
		else:
			return self.moco_forward(*args, prints=prints)
	def feature_forward(self, q):
		q_enc = self.encoder_q(q, True)  # queries: NxC
		return q_enc
	def moco_forward(self, q, k, prints=False):
		""" moco phase forward pass """
		print('q in', q.shape) if prints else None
		print('k in', k.shape) if prints else None

		q_enc = self.encoder_q(q)  # queries: NxC
		q = self.q_fc(q_enc)
		q = nn.functional.normalize(q, dim=1)
		print('q_encoder(q)', q.shape) if prints else None

		with torch.no_grad():
			k = self.encoder_k(k)  # keys: NxC
			k = self.k_fc(k)
			k = nn.functional.normalize(k, dim=1)
		print('k_encoder(k)', k.shape) if prints else None

		# positive logits: Nx1
		l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
		print('l_pos', l_pos.shape) if prints else None

		# negative logits: NxK
		print('self.queue', self.queue.shape) if prints else None
		l_neg = torch.einsum('nc,kc->nk', [q, self.queue.clone().detach()])
		print('l_neg', l_neg.shape) if prints else None

		# logits: Nx(1+K)
		logits = torch.cat([l_pos, l_neg], dim=1) / self.T
		print('logits', logits.shape) if prints else None

		# contrastive loss labels, positive logits used as ground truth
		zeros = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
		print('zeros', zeros.shape) if prints else None

		self.update_k_encoder_weights()
		self.update_queue(k)

		return logits, zeros


#Barlow Twins
class Barlow_Twins(nn.Module):
	def __init__(self,base_encoder, dim=128):
		super(Barlow_Twins, self).__init__()

		# encoder
		self.f = base_encoder(num_classes= dim)
		# projection head
		self.g = nn.Sequential(nn.Linear(self.f.fc.weight.shape[1], self.f.fc.weight.shape[1], bias=False), nn.BatchNorm1d(self.f.fc.weight.shape[1]),
							   nn.ReLU(inplace=True), nn.Linear(self.f.fc.weight.shape[1], dim, bias=True))
		self.f_fc = self.f.fc
		self.f.fc = Dummy()

	def forward(self, x, feature_only=False):
		if feature_only:
			return self.f(x, True)
		else:
			x_f = self.f(x)
			out = self.g(x_f)
			return F.normalize(x_f, dim=-1), F.normalize(out, dim=-1)

#SimCLR
class SimCLR(nn.Module):
	def __init__(self,base_encoder, dim=128):
		super(SimCLR, self).__init__()

		# encoder
		self.f = base_encoder(num_classes= dim)
		# projection head
		self.g = nn.Sequential(nn.Linear(self.f.fc.weight.shape[1], self.f.fc.weight.shape[1], bias=False), nn.BatchNorm1d(self.f.fc.weight.shape[1]),
							   nn.ReLU(inplace=True), nn.Linear(self.f.fc.weight.shape[1], dim, bias=True))
		self.f_fc = self.f.fc
		self.f.fc = Dummy()

	def forward(self, x, feature_only=False):
		if feature_only:
			return self.f(x, True)
		else:
			x_f = self.f(x)
			out = self.g(x_f)
			return F.normalize(x_f, dim=-1), F.normalize(out, dim=-1)

#SwAV
class SwAV(nn.Module):
	def __init__(
			self,
			base_encoder, dim=128,
			normalize=True,
			nmb_prototypes=3000,
			eval_mode=False,
	):
		super(SwAV, self).__init__()

		self.eval_mode = eval_mode

		self.f = base_encoder(num_classes= dim)
		# normalize output features
		self.l2norm = normalize

		# projection head
		self.projection_head = nn.Sequential(
				nn.Linear(self.f.fc.weight.shape[1], self.f.fc.weight.shape[1]),
				nn.BatchNorm1d(self.f.fc.weight.shape[1]),
				nn.ReLU(inplace=True),
				nn.Linear(self.f.fc.weight.shape[1], dim),
			)

		self.f_fc = self.f.fc
		self.f.fc = Dummy()
		# prototype layer
		self.prototypes = None
		self.prototypes = nn.Linear(dim, nmb_prototypes, bias=False)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward_backbone(self, x):
		x = self.f(x)
		return x

	def forward_head(self, x):
		if self.projection_head is not None:
			x = self.projection_head(x)

		if self.l2norm:
			x = nn.functional.normalize(x, dim=1, p=2)

		if self.prototypes is not None:
			return x, self.prototypes(x)
		return x

	def forward(self, inputs, feature_only=False):
		if feature_only:
			return self.f(inputs, True)
		else:
			if not isinstance(inputs, list):
				inputs = [inputs]
			idx_crops = torch.cumsum(torch.unique_consecutive(
				torch.tensor([inp.shape[-1] for inp in inputs]),
				return_counts=True,
			)[1], 0)
			start_idx = 0
			for end_idx in idx_crops:
				_out = self.forward_backbone(torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True))
				if start_idx == 0:
					output = _out
				else:
					output = torch.cat((output, _out))
				start_idx = end_idx
			return self.forward_head(output)

