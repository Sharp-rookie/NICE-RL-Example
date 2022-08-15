import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

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


################################## PPO Policy ##################################
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = [] # log[p(a_t|s_t)]，用于计算重要性因子
        self.action_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.action_values[:]
        del self.is_terminals[:]


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
        """执行act与环境交互的的是 agent'，也就是θ’而不是θ
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state) # 对于策略梯度，actor网络给出的是可选动作的概率分布，在这里就是每个动作对应正态分布的均值
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # 每个动作的正态分布标准差
            dist = MultivariateNormal(action_mean, cov_mat) # 连续动作采用多元正态分布，目的是逐渐逼近真正的动作选择的正态分布
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample() # 从给定策略对应的动作概率分布中采样获得本次动作
        action_logprob = dist.log_prob(action) # 选择此动作对应的概率的log，用于计算重要性因子
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        """执行evaluate从而进行训练更新的是 agent，也就是θ
        """
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() # 动作分布的熵值，用于鼓励探索
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    """这里是PPO-clip算法
    """

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.memory = Memory()

        # 用于训练学习的agent θ
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # 用于与环境交互的agent θ’
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """连续动作的正态分布标准差衰减
        """
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
        """agent θ’ 与环境交互，选择动作并计入memory
        """
        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.memory.states.append(state)
            self.memory.actions.append(action)
            self.memory.logprobs.append(action_logprob)

            return action.item()

    def update(self):
        """agent θ 利用观测内容训练更新 θ 参数
        """
        # Monte Carlo法 估计动作价值 action_values
        action_values = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.action_values), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            action_values.insert(0, discounted_reward)
            
        # Normalizing the action_values
        action_values = torch.tensor(action_values, dtype=torch.float32).to(device)
        action_values = (action_values - action_values.mean()) / (action_values.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).detach().to(device)

        # agent θ’与环境交互一波后，agent θ 用获取的memory数据训练更新 K 轮
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # 计算重要性因子 (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 按PPO-clip的公式计算策略梯度
            advantages = action_values - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 这里总的loss = Actor网络的策略梯度 + Critic网络的MSE损失 + Actor选择动作的熵值（鼓励探索）
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, action_values) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear memory
        self.memory.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       


