import torch
import numpy as np
import minari
from policy import FlowPolicy
import time


# ======================================
# ODE SAMPLER FOR FLOW POLICY
# ======================================
def sample_action(policy, state, action_dim, steps=20):
    with torch.no_grad():
        a = torch.randn(1, action_dim).to(state.device)
        dt = 1.0 / steps

        # integrate from t=1 → 0
        for i in reversed(range(steps)):
            t = torch.ones(1, 1).to(state.device) * (i / steps)
            v = policy(a, t, state)
            a = a - v * dt

        return torch.tanh(a)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("flow_halfcheetah.pt", map_location=device)

    state_dim = 17
    action_dim = 6

    policy = FlowPolicy(state_dim, action_dim).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()

    obs_mean = checkpoint["obs_mean"].to(device)
    obs_std = checkpoint["obs_std"].to(device)

    dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")
    env = dataset.recover_environment(eval_env=True, render_mode="human")

    num_episodes = 3

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            obs_norm = (obs_tensor - obs_mean) / obs_std

            action_tensor = sample_action(policy, obs_norm, action_dim, steps=20)
            action = action_tensor.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            time.sleep(0.01)

        print(f"Episode {ep+1} | Return: {total_reward:.2f}")

    env.close()