import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_dim)
        
        self.max_action = max_action

        self.apply(init_weights)
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a
        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 1)

        self.apply(init_weights)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], -1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
class DDPG:
    def __init__(self, lr, state_dim, action_dim, max_action):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.max_action = max_action
    
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak):
        
        for _ in range(n_iter):
            
            ####################
            # Sample a batch
            ####################
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)

            ####################
            # Actor Loss
            ####################
            actor_loss = -self.critic(state, self.actor(state)).mean()

            ####################
            # Critic Loss
            ####################
            # Select next action according to target actor with noise:
            next_action = (self.actor_target(next_state)).clamp(-self.max_action, self.max_action)

            # target Q-value
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()

            # current Q-value
            current_Q = self.critic(state, action)
            loss_Q1 = F.mse_loss(current_Q, target_Q)

            ####################
            # Optimization
            ####################
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_optimizer.step()

            ####################
            # Polyak averaging update
            ####################
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                    
                
    def save(self, checkpoint_path):
        torch.save(self.actor.state_dict(), checkpoint_path+'_actor.pth')
        torch.save(self.actor_target.state_dict(), checkpoint_path+'_actor_target.pth')
        
        torch.save(self.critic.state_dict(), checkpoint_path+'_crtic.pth')
        torch.save(self.critic_target.state_dict(), checkpoint_path+'_critic_target.pth')
        
    def load(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path+'_actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load(checkpoint_path+'_actor_target.pth', map_location=lambda storage, loc: storage))
        
        self.critic.load_state_dict(torch.load(checkpoint_path+'_crtic.pth', map_location=lambda storage, loc: storage))
        self.critic_target.load_state_dict(torch.load(checkpoint_path+'_critic_target.pth', map_location=lambda storage, loc: storage))
        
    def load_actor(self, checkpoint_path):
        self.actor.load_state_dict(torch.load(checkpoint_path+'_actor.pth', map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load(checkpoint_path+'_actor_target.pth', map_location=lambda storage, loc: storage))