from PIL import Image
import numpy as np
from heapq import heappop, heappush
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import network
from Env import StaticEnvironment
from Agent import nn_Agent


if __name__ == '__main__':
    # GPU设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    save_path = "models/model_test.pkl"
    load_path = "models/1221_14fov_3_80reached.pkl"
    # 环境和智能体初始化
    env = StaticEnvironment(local_fov=14, num_dyna=50)
    # env.is_render = True
    state = env.reset()
    model = network.CNN(env.dim_states, env.n_actions).to(device)
    # model.load_state_dict(torch.load(load_path))
    agent = nn_Agent(env, model, device)
    # 测试
    input = torch.from_numpy(state.T).float().to(device)
    # print(input.size())
    # 训练参数设置
    batch_size = 5
    num_of_episodes = 1000
    timesteps_per_episode = 1000
    
    num_finished_episodes = 0
    # 训练
    reward_list = []
    loss_list = []
    step_list = []
    for i in trange(num_of_episodes):
        loss_episodes = 0
        reward_episodes = 0
        times = 0
        # Reset the enviroment
        state = env.reset()
        for timestep in range(timesteps_per_episode):
            # Run Action
            input = torch.from_numpy(state.T).float().to(device)
            # import pdb; pdb.set_trace()
            action = agent.act(input)
            # Take action    
            reward, next_state, done = env.step(action)
            reward_episodes += reward
            # Store the transition
            agent.store(state, action, reward, next_state, done)
            # Retrain the agent
            if len(agent.expirience_replay) >= batch_size:
                loss = agent.retrain(batch_size)
                loss_episodes += loss
                times += 1
            # Update the state
            state = next_state
            # Update the plot
            if done:
                break
        # Update the target network with new weights
        agent.alighn_target_model()
        if timestep < timesteps_per_episode - 1:
            num_finished_episodes += 1
        print("time_step:", times, "loss:", loss_episodes/times)
        loss_list.append(loss_episodes/times)
        reward_list.append(reward_episodes/timestep)
        step_list.append(timestep)
    print("num_finished_episodes:", num_finished_episodes, "percentage:", num_finished_episodes/num_of_episodes)
    torch.save(agent.q_network.state_dict(), save_path)
    add_str = "_0410_1"
    np.save("plot_data/loss"+add_str, np.array(loss_list))
    np.save("plot_data/reward"+add_str, np.array(reward_list))
    np.save("plot_data/steps"+add_str, np.array(step_list))
