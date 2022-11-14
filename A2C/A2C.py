import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

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


########################### Monte Carlo法估计动作价值 ###########################
def compute_returns(next_value, rewards, is_terminals, gamma=0.99):
    
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1-int(is_terminals[step]))
        returns.insert(0, R)
    
    return returns


################################## A2C Policy ##################################
class Memory:
    def __init__(self):
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = [] # PPO 不需要存current critic的状态价值，因为更新时会用target critic现场计算状态价值
        self.is_terminals = []
        self.entropy = 0
    
    def clear(self):
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        self.entropy = 0

def init_weights(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
         torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, torch.nn.Conv2d):
         torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
        self.apply(init_weights)
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        
        else: # 只有连续动作才有概率分布，从而才有方差；离散动作只有自身的绝对概率值
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        """决策"""
        
        if self.has_continuous_action_space:
            action_mean = self.actor(state) # 对于策略梯度，actor网络给出的是可选动作的概率分布，在这里就是每个动作对应正态分布的均值
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # 每个动作的正态分布标准差
            dist = MultivariateNormal(action_mean, cov_mat) # 连续动作采用多元正态分布，目的是逐渐逼近真正的动作选择的正态分布
        
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample() # 从给定策略对应的动作概率分布中采样获得本次动作
        action_logprob = dist.log_prob(action) # 选择此动作对应的概率的log，用于计算重要性因子
        action_entropy = dist.entropy() # 动作分布的熵值，用于鼓励探索
        
        return action, action_logprob, action_entropy
    
    def evaluate(self, state):
        """评估"""

        state_values = self.critic(state)
        return state_values


class A2C:

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.memory = Memory()

        # 用于训练学习的agent θ
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """连续动作的正态分布标准差衰减"""

        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):
        """agent与环境交互，选择动作并计入memory"""

        state = torch.FloatTensor(state).to(device)
        action, action_logprob, action_entropy = self.policy.act(state)

        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob)
        self.memory.entropy += action_entropy
        self.memory.state_values.append(self.policy.evaluate(state))

        return action.cpu().numpy().flatten() if self.has_continuous_action_space else action.item()

    def update(self, next_state):
        """agent利用观测内容训练更新参数"""

        # 蒙特卡洛法计算动作价值
        next_state = torch.FloatTensor(next_state).to(device)
        next_value = self.policy.evaluate(next_state)
        action_values = compute_returns(next_value, self.memory.rewards, self.memory.is_terminals)
        
            
        # Normalizing the action_values
        action_values = torch.tensor(action_values, dtype=torch.float32).to(device)
        action_values = (action_values - action_values.mean()) / (action_values.std() + 1e-7)

        # Convert list to tensor
        logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).to(device)

        state_values = torch.FloatTensor(self.memory.state_values).to(device)
        advantages = action_values - state_values
        actor_loss  = -(logprobs * advantages.detach()).mean()
        critic_loss = self.MseLoss(state_values, action_values).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * self.memory.entropy

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # clear memory
        self.memory.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))