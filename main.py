if __name__ == "__main__":
    import torch
    import minari
    import numpy as np

    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    obs_list, action_list = [], []
    next_obs_list, reward_list, done_list = [], [], []

    for episode in dataset.iterate_episodes():
        obs_ep = episode.observations          # shape (T+1, obs_dim)
        act_ep = episode.actions               # shape (T, action_dim)
        rew_ep = episode.rewards               # shape (T,)
        term_ep = episode.terminations         # shape (T,)
        trunc_ep = episode.truncations         # shape (T,)

        # transitions within this episode
        for t in range(len(act_ep)):
            obs_list.append(obs_ep[t])
            action_list.append(act_ep[t])
            next_obs_list.append(obs_ep[t+1])
            reward_list.append(rew_ep[t])
            done_list.append(bool(term_ep[t] or trunc_ep[t]))

    # Convert lists of numpy arrays to single numpy arrays
    obs_np = np.array(obs_list)
    actions_np = np.array(action_list)
    next_obs_np = np.array(next_obs_list)
    rewards_np = np.array(reward_list).reshape(-1, 1)
    dones_np = np.array(done_list).reshape(-1, 1)

    # Convert to tensors
    obs = torch.from_numpy(obs_np).float()
    actions = torch.from_numpy(actions_np).float()
    next_obs = torch.from_numpy(next_obs_np).float()
    rewards = torch.from_numpy(rewards_np).float()
    dones = torch.from_numpy(dones_np).float()

    print("Obs:", obs.shape)
    print("Actions:", actions.shape)
    print("Next obs:", next_obs.shape)
    print("Rewards:", rewards.shape)
    print("Dones:", dones.shape)


    from torch.utils.data import TensorDataset, DataLoader

    dataset_torch = TensorDataset(obs, actions, next_obs, rewards, dones)
    dataloader = DataLoader(dataset_torch, batch_size=256, shuffle=True)