import gym
import time
import time
import visdom
import numpy as np

from PPO import PPO


class Visualizer(object):
    """
    封装visdom的基本操作
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        self.index = {}
        self.log_text = ''

    def reinit(self, env="default", **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        """
        一次img_many
        @params d: dict (name, value) i.e. ('image', img_tensor)
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, win, name, y, **kwargs):
        """
        self.plot('loss', 1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=win,
            name=name,
            opts=dict(
                title=win,
                showlegend=False,  # 显示网格
                xlabel='x1',  # x轴标签
                ylabel='y1',  # y轴标签
                fillarea=False,  # 曲线下阴影覆盖
                width=2400,  # 画布宽
                height=350,  # 画布高
            ),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img, torch.Tensor(64,64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)

        !!! don't ~~self.img('input_imgs', t.Tensor(100, 64, 64), nrows=10)~~ !!!
        """
        self.vis.images(img_.cpu().numpy(), win=name,
                        opts=dict(title=name), **kwargs)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1, 'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        """
        自定义的plot,image,log,plot_many等除外
        self.function 等价于self.vis.function
        """
        return getattr(self.vis, name)


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################
    env_name = "RoboschoolWalker2d-v1"
    has_continuous_action_space = True
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    use_vis = False
    vis_env = env_name
    port = 8097
    #####################################################

    env = gym.make(env_name)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    # visualization
    if use_vis:
        vis = Visualizer(vis_env, port=port)

    step = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

            step += 1
            
            # Visualize
            if step % 1 == 0:
                vis.plot(win='Offset', name='GBR', y=env.bucket.offset)
                vis.plot(win='reservedPRB', name='GBR', y=env.bucket.reservedPRB)
                vis.plot(win='Reward', name='GBR', y=reward)
                vis.plot(win='Delay', name='GBR', y=env.bucket.delay)

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
