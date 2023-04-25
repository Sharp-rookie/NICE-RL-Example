import os
import torch
import gym
import numpy as np

from TD3 import TD3, ReplayBuffer

def train():
    ######### Hyperparameters #########
    env_name = "BipedalWalker-v2"
    log_interval = 10           # print avg reward after interval
    save_interval = 100
    random_seed = 0
    gamma = 0.99                # discount for future rewards
    batch_size = 100            # num of transitions sampled from replay buffer
    lr = 0.001
    exploration_noise = 0.6                  # starting noise for action
    exploration_noise_min = 0.1              # minimum action_noise
    exploration_noise_decay_step = 0.05      # linearly decay action_noise
    exploration_noise_decay_interval = 50   # action noise decay interval
    polyak = 0.995              # target policy update parameter (1-tau)
    policy_noise = 0.2          # target policy smoothing noise
    noise_clip = 0.5
    policy_delay = 2            # delayed policy updates parameter
    max_episodes = 1000         # max num of episodes
    max_timesteps = 500        # max timesteps in one episode
    checkpoint_path = "./preTrained/{}/".format(env_name) # save trained models
    ###################################

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    policy = TD3(lr, state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(max_timesteps*10)
    
    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    # logging variables:
    time_steps = 0
    avg_reward = 0
    ep_reward = 0
    log_f = open("log.txt","w+")
    log_f.write('episode,time_steps,ep_reward\n')
    
    # training procedure:
    for episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            
            # take action in env:
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            
            avg_reward += reward
            ep_reward += reward
            
            # if time_steps >= batch_size:
            #     policy.update(replay_buffer, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            
            time_steps += 1

            if done:
                break

            env.render()
        
        policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
        
        # logging updates:
        log_f.write('{},{},{}\n'.format(episode, time_steps, ep_reward))
        log_f.flush()
        ep_reward = 0

        if (episode+1) % exploration_noise_decay_interval==0 and exploration_noise>exploration_noise_min:
            exploration_noise = round(exploration_noise - exploration_noise_decay_step, 2)
            print(f'exploration noise decay to {exploration_noise}')
        
        if episode % save_interval == 0:
            policy.save(checkpoint_path+str(episode))
        
        # print avg reward every log interval:
        if episode % log_interval == 0:
            avg_reward = avg_reward / log_interval
            print("Episode: {}\tAverage Reward: {:.2f}".format(episode, avg_reward))
            avg_reward = 0

if __name__ == '__main__':
    train()
    
