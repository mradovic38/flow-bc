import torch
import numpy as np
import minari
from policy import GaussianPolicy
import time


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("bc_gaussian_halfcheetah.pt", map_location=device)
    state_dim = 17
    action_dim = 6

    policy = GaussianPolicy(state_dim, action_dim).to(device)
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
            # Normalize observation
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            obs_norm = (obs_tensor - obs_mean) / obs_std

            with torch.no_grad():
                dist = policy(obs_norm)
                action = dist.sample().cpu().numpy()[0]
                action = np.tanh(action)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            time.sleep(0.01)

        print(f"Episode {ep+1} | Return: {total_reward:.2f}")

    env.close()