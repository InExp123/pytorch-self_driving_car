import torch as T 
import torch.nn as nn 
import numpy as np 
import sys
from DQnet import DeepQNetwork
from buffer import Replay_mem
dtype = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor
from settings import IM_HEIGHT, IM_WIDTH, FPS
import time

T.manual_seed(0)

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=40000, esp_end = 0.01, eps_dec = 5e-4,\
        save_targ_net = 200, rl_algo = None, env_name = None, chkpt_dir='tmp'): 
        self.rl_algo = rl_algo
        self.env_name = env_name
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.action_space = [i for i in range(n_actions)]
        self.eps_min = esp_end
        self.eps_dec = eps_dec
        self.learn_step_counter = 0
        self.repeat_save = save_targ_net

        self.Q_local =DeepQNetwork(self.lr, n_actions, chkpt_dir,\
                                    self.env_name + self.rl_algo + 'Q_local')
        self.Q_target =DeepQNetwork(self.lr, n_actions, chkpt_dir,\
                                    self.env_name + self.rl_algo + 'Q_target')

        self.replay_mem = Replay_mem(max_mem_size, input_dims, batch_size)
        self.stop = False
        self.training_initialized = False

    def store_transition(self, state, action, reward, state_, done):
        self.replay_mem.store_transition(state, action, reward, state_, done)
        

    def choose_actions(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
            time.sleep(1/FPS)

        else:
            actions = self.Q_local.forward(observation)
            action = T.argmax(actions).item()

        return action

    def replace_target_network(self):
        if self.learn_step_counter % self.repeat_save == 0:
            self.Q_target.load_state_dict(self.Q_local.state_dict())

    def reduce_eps(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.replay_mem.mem_cntr < self.replay_mem.mem_size:
            # if (self.replay_mem.mem_cntr%100 ==0):
            #     print(self.replay_mem.mem_cntr)
            return
        self.Q_local.optimizer.zero_grad()
        self.replace_target_network()
        
        state_batch, next_state_batch, reward_batch, terminal_batch, action_batch = self.replay_mem.sample()

        state_batch = T.tensor(state_batch).to(self.Q_local.device)
        next_state_batch = T.tensor(next_state_batch).to(self.Q_local.device)
        reward_batch = T.tensor(reward_batch).to(self.Q_local.device)
        terminal_batch = T.tensor(terminal_batch).bool().to(self.Q_local.device)
        action_batch = T.tensor(action_batch).long().to(self.Q_local.device)

        q_eval = self.Q_local.forward(state_batch).gather(1, action_batch.unsqueeze(1)).view(self.batch_size)
        qt_next = self.Q_target.forward(next_state_batch)
        ql_next = self.Q_local.forward(next_state_batch)

        max_actions = T.argmax(ql_next, dim=1)
        # print("qt_next: ",qt_next)
        # print(q_next.shape)
        qt_next[terminal_batch] = 0
        # print("qt_next: ",qt_next)
        # print(q_next.shape)
        # print('terminal_batch, ', terminal_batch)
        # print(terminal_batch.shape)
        # sys.exit("Exit from here")
        q_target = reward_batch + self.gamma*(qt_next.gather(1,max_actions.unsqueeze(1)).view(self.batch_size))
        
        #sys.exit("Exit from here")

        loss = self.Q_local.loss(q_target,q_eval).to(self.Q_local.device)
        loss.backward()
        self.Q_local.optimizer.step()
        
        self.learn_step_counter += 1
        self.reduce_eps()

    def train_in_loop(self):
        x = np.random.uniform(size=(1, 3, IM_HEIGHT, IM_WIDTH)).astype(np.float32)
        # y = np.random.uniform(size=(1, 3)).astype(np.float32)

        q_eval = self.Q_local.forward(x).detach().cpu().numpy()
        
        self.Q_target.load_state_dict(self.Q_local.state_dict())


        self.training_initialized = True
        print("p1")

        while True:
            if self.stop:
                return
            self.learn()
            time.sleep(0.01)

    def save_models(self):
        self.Q_local.save_checkpoint()
        self.Q_target.save_checkpoint()

    def load_models(self):
        self.Q_local.load_checkpoint()
        self.Q_target.load_checkpoint()

        