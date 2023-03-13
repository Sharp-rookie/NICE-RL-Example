# 经典RL算法

| 算法        | 类型               | 确定性 | 离散 | 连续 | 使用体验                   |
| ----------- | ------------------ | ------ | ---- | ---- | -------------------------- |
| Nature DQN  | VB  \|  off-policy | No     | Yes  | No   | toy model，没人用          |
| Double DQN  | VB  \|off-policy   | No     | Yes  | No   | to do                      |
| Dueling DQN | VB  \|off-policy   | No     | Yes  | No   | to do                      |
| Rainbow DQN | TODO               | No     | Yes  | No   | to do                      |
| A2C         | PG  \|  on-policy  | No     | Yes  | Yes  | 评价是，不如 PPO           |
| A3C         | PG  \|  on-policy  | No     | Yes  | Yes  | 多进程并行，硬件设备要求高 |
| DDPG        | PG  \|  off-policy | Yes    | Yes  | Yes  | 评价是，不如 TD3           |
| TD3         | PG  \|  off-policy | Yes    | Yes  | Yes  | 训练慢，难调参             |
| PPO         | PG  \|  on-policy  | No     | Yes  | Yes  | 训练快，好调参，效果好     |
| SAC         | PG  \|off-policy   | No     | Yes  | Yes  | 可能比TD3还难训            |



**Reference：**

[spinningup/spinup/algos/pytorch](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch)

[higgsfield/RL-Adventure-2](https://github.com/higgsfield/RL-Adventure-2)

[easy-rl/projects/codes](https://github.com/datawhalechina/easy-rl/tree/master/projects/codes)

[Official TD3 Code](https://github.com/sfujim/TD3/)

[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)

[SAC Algorithm](https://zhuanlan.zhihu.com/p/385658411)