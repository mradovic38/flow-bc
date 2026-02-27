import torch
import torch.nn as nn
import minari
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from policy import FlowPolicy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    obs_list, action_list = [], []

    for episode in dataset.iterate_episodes():
        obs_ep = episode.observations
        act_ep = episode.actions

        for t in range(len(act_ep)):
            obs_list.append(obs_ep[t])
            action_list.append(act_ep[t])

    obs = torch.from_numpy(np.array(obs_list)).float()
    actions = torch.from_numpy(np.array(action_list)).float()

    print("Obs:", obs.shape)
    print("Actions:", actions.shape)

    # Normalize observations
    obs_mean = obs.mean(0, keepdim=True)
    obs_std = obs.std(0, keepdim=True) + 1e-6

    obs = (obs - obs_mean) / obs_std

    dataset_torch = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset_torch, batch_size=256, shuffle=True)

    state_dim = obs.shape[1]
    action_dim = actions.shape[1]

    policy = FlowPolicy(state_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # ==============================
    # FLOW MATCHING LOSS
    # ==============================
    def flow_matching_loss(policy, states, actions):
        batch_size = states.size(0)

        x0 = actions
        x1 = torch.randn_like(x0)

        t = torch.rand(batch_size, 1, device=states.device)

        xt = (1 - t) * x0 + t * x1

        target_velocity = x1 - x0

        pred_velocity = policy(xt, t, states)

        loss = ((pred_velocity - target_velocity) ** 2).mean()
        return loss

    # Training
    epochs = 100

    for epoch in range(epochs):
        total_loss = 0

        for states, acts in dataloader:
            states = states.to(device)
            acts = acts.to(device)

            loss = flow_matching_loss(policy, states, acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:03d} | Loss: {total_loss / len(dataloader):.6f}")

    torch.save({
        "policy": policy.state_dict(),
        "obs_mean": obs_mean,
        "obs_std": obs_std
    }, "flow_halfcheetah.pt")