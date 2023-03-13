import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim

import warnings
warnings.simplefilter('ignore')

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## DDPG Policy ##################################
class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size = (self.size+1) % self.max_size
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
         torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, torch.nn.Conv2d):
         torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

        self.apply(init_weights)
        
    def forward(self, state):
        out = self.net(state)
        return out
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SoftQNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(num_inputs + num_actions, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 1)
        )

        self.apply(init_weights)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        out = self.net(x)
        return out
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, max_action, log_std_min=-1, log_std_max=1):
        super(PolicyNetwork, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.backbone = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.Tanh(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.LayerNorm(64)
        )
        
        self.mean_linear = nn.Linear(64, num_actions)
        self.log_std_linear = nn.Linear(64, num_actions)

        self.apply(init_weights)
        
    def forward(self, state):

        x = self.backbone(state)
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        action = action * self.max_action
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        
        action  = action * self.max_action
        return action.flatten()
    

class SAC():
    def __init__(self, lr, state_dim, action_dim, max_action):

        self.policy = PolicyNetwork(state_dim, action_dim, max_action, log_std_min=-1, log_std_max=1).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.value_net = ValueNetwork(state_dim).to(device)
        self.value_net_target = ValueNetwork(state_dim).to(device)
        self.value_optimizer  = optim.Adam(self.value_net.parameters(), lr=lr)

        self.soft_q_net1 = SoftQNetwork(state_dim, action_dim).to(device)
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)

        self.soft_q_net2 = SoftQNetwork(state_dim, action_dim).to(device)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy.get_action(state).cpu().data.numpy().flatten()

    def update(self, replay_buffer, batch_size, gamma, polyak, mean_lambda=1e-3, std_lambda=1e-3, z_lambda=0.0):

        ####################
        # Sample a batch
        ####################
        state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action_).to(device)
        reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)

        expected_value   = self.value_net(state)
        new_action, log_prob, z, mean, log_std = self.policy.evaluate(state)

        ####################
        # Soft Q Loss
        ####################
        expected_q_value1 = self.soft_q_net1(state, action)
        expected_q_value2 = self.soft_q_net2(state, action)
        target_value = self.value_net_target(next_state)
        next_q_value = reward + (1 - done) * gamma * target_value
        q_value_loss1 = F.mse_loss(expected_q_value1, next_q_value.detach())
        q_value_loss2 = F.mse_loss(expected_q_value2, next_q_value.detach())

        ####################
        # Value Loss
        ####################
        expected_min_q_value = torch.min(self.soft_q_net1(state, new_action), self.soft_q_net2(state, new_action))
        next_value = expected_min_q_value - log_prob
        value_loss = F.mse_loss(expected_value, next_value.detach())

        ####################
        # Actor(Policy) Loss
        ####################
        expected_q_value = self.soft_q_net1(state, action)
        log_prob_target = expected_q_value - expected_value
        policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
        
        mean_loss = mean_lambda * mean.pow(2).mean()
        std_loss  = std_lambda  * log_std.pow(2).mean()
        z_loss    = z_lambda    * z.pow(2).sum(1).mean()

        policy_loss += mean_loss + std_loss + z_loss

        ####################
        # Optimization
        ####################
        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
    
        ####################
        # Polyak averaging update
        ####################
        for target_param, param in zip(self.value_net_target.parameters(), self.value_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - polyak) + param.data * polyak)
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path+'_policy.pth')
        
        torch.save(self.value_net.state_dict(), checkpoint_path+'_value_net.pth')
        torch.save(self.value_net_target.state_dict(), checkpoint_path+'_value_net_target.pth')
        
        torch.save(self.soft_q_net1.state_dict(), checkpoint_path+'_soft_q_net1.pth')
        torch.save(self.soft_q_net2.state_dict(), checkpoint_path+'_soft_q_net2.pth')
        
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path+'_policy.pth', map_location=lambda storage, loc: storage))
        
        self.value_net.load_state_dict(torch.load(checkpoint_path+'_value_net.pth', map_location=lambda storage, loc: storage))
        self.value_net_target.load_state_dict(torch.load(checkpoint_path+'_value_net_target.pth', map_location=lambda storage, loc: storage))
        
        self.soft_q_net1.load_state_dict(torch.load(checkpoint_path+'_soft_q_net1.pth', map_location=lambda storage, loc: storage))
        self.soft_q_net2.load_state_dict(torch.load(checkpoint_path+'_soft_q_net2.pth', map_location=lambda storage, loc: storage))
        
    def load_policy(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))