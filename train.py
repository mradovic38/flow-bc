import torch
import torch.nn as nn
import minari
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from policy import GaussianPolicy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    obs_list, action_list = [], []
    next_obs_list, reward_list, done_list = [], [], []

    for episode in dataset.iterate_episodes():
        obs_ep = episode.observations
        act_ep = episode.actions
        rew_ep = episode.rewards
        term_ep = episode.terminations
        trunc_ep = episode.truncations

        for t in range(len(act_ep)):
            obs_list.append(obs_ep[t])
            action_list.append(act_ep[t])
            next_obs_list.append(obs_ep[t+1])
            reward_list.append(rew_ep[t])
            done_list.append(bool(term_ep[t] or trunc_ep[t]))

    obs = torch.from_numpy(np.array(obs_list)).float()
    actions = torch.from_numpy(np.array(action_list)).float()
    next_obs = torch.from_numpy(np.array(next_obs_list)).float()
    rewards = torch.from_numpy(np.array(reward_list)).float().unsqueeze(1)
    dones = torch.from_numpy(np.array(done_list)).float().unsqueeze(1)

    print("Obs:", obs.shape)
    print("Actions:", actions.shape)

    # Normalize observations
    obs_mean = obs.mean(0, keepdim=True)
    obs_std = obs.std(0, keepdim=True) + 1e-6

    obs = (obs - obs_mean) / obs_std
    next_obs = (next_obs - obs_mean) / obs_std

    dataset_torch = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset_torch, batch_size=256, shuffle=True)

    state_dim = obs.shape[1]
    action_dim = actions.shape[1]

    policy = GaussianPolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # NLL
    def bc_loss(policy, states, actions):
        dist = policy(states)

        eps = 1e-6
        atanh_actions = torch.atanh(actions.clamp(-1 + eps, 1 - eps))

        log_prob = dist.log_prob(atanh_actions).sum(-1)
        return -log_prob.mean()

    # Training
    epochs = 50

    for epoch in range(epochs):
        total_loss = 0

        for states, acts in dataloader:
            states = states.to(device)
            acts = acts.to(device)

            loss = bc_loss(policy, states, acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | Loss: {total_loss / len(dataloader):.4f}")

    
    torch.save({
        "policy": policy.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std
    }, "bc_gaussian_halfcheetah.pt")