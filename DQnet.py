import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import numpy as np 
import sys
import os
from torchsummary import summary
dtype = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor

class CNN64x3(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(CNN64x3, self).__init__()
		self.conv = nn.Conv2d(in_channels=input_channels, kernel_size=3, out_channels=output_channels)
		self.relu = nn.ReLU()
		self.pool = nn.AvgPool2d(5, stride=3, padding=0)
	
	def forward(self, batch_data):
		output = self.conv(batch_data)
		output = self.relu(output)
		output = self.pool(output)

		return output

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = T.prod(T.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

class DeepQNetwork(nn.Module):
	def __init__(self, lr, n_actions, chkpt_dir, name, input_channels=3):
		super(DeepQNetwork, self).__init__()
		self.checkpoint_dir = chkpt_dir
		self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
		self.lr = lr
		self.n_actions = n_actions
		self.cnn64_1 = CNN64x3(input_channels, 64)
		self.cnn64_2 = CNN64x3(64, 64)
		self.cnn64_3 = CNN64x3(64, 64)

		self.fc = nn.Linear(64*21*15, self.n_actions) #line 32 could also be tried

		self.network = nn.Sequential(self.cnn64_1, self.cnn64_2, self.cnn64_3)
		
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()

		# self.device = T.device('cuda') if T.cuda.is_available() else T.device('cpu')
		self.device = T.device('cuda')
		self.to(self.device)

	def forward(self, observation):
		state = T.as_tensor(observation,dtype=T.float32).to(self.device)
		x = self.network(state)
		x = x.view(x.size()[0], -1)
		x = self.fc(x)

		return x

	def save_checkpoint(self):
		print('... saving checkpoint ...')
		T.save(self.state_dict(), self.checkpoint_file)
		
	def load_checkpoint(self):
		print('... loading checkpoint ...')
		self.load_state_dict(T.load(self.checkpoint_file))



model = DeepQNetwork(0.0001, 3, "test", "test")
x = T.randn(3, 640, 480)
x = np.expand_dims(x, axis=0)
print(x.shape)
# summary(model, (3, 640, 480))
print(model.forward(x).detach().cpu().numpy())
