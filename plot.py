import pandas as pd
import matplotlib.pyplot as plt

a2c = pd.read_csv('A2C\A2C_logs\BipedalWalker-v2\A2C_BipedalWalker-v2_log_2.csv')
ppo = pd.read_csv('PPO\PPO_logs\BipedalWalker-v2\PPO_BipedalWalker-v2_log_4.csv')
# td3 = pd.read_csv('')
# ddpg = pd.read_csv('')

plt.figure(figsize=(21,7))

plt.plot(a2c['timestep'], a2c['reward'], label='A2C')
plt.plot(ppo['timestep'], ppo['reward'], label='PPO')
# plt.plot(td3['timestep'], td3['reward'], label='TD3')
# plt.plot(ddpg['timestep'], ddpg['reward'], label='DDPG')

plt.legend()
plt.savefig('comp.png', dpi=400)