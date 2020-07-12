import torch as T 
import torch.nn as nn 
import numpy as np 

class Replay_mem():
    def __init__(self, mem_size, input_dims, batch_size):
        self.batch_size = batch_size
        self.mem_size = mem_size
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.nex_state = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_cntr = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr%self.mem_size
        self.state_memory[index] = state
        self.nex_state[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        return self.state_memory[batch], self.nex_state[batch], \
            self.reward_memory[batch], self.terminal_memory[batch], self.action_memory[batch]