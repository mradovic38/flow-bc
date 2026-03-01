import os
import argparse
import datetime
import yaml

import numpy as np
import torch
import minari
from gymnasium.wrappers import RecordVideo

from policy import GaussianPolicy, FlowPolicy

DEVICE="cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC model from YAML config with video recording")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint override")
    parser.add_argument("--video-dir", type=str, default="./videos", help="Directory to save videos")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to run")

    return parser.parse_args()


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    params = {}
    for k, v in cfg.get("parameters", {}).items():
        if "value" in v:
            params[k.replace("-", "_")] = v["value"]
        elif "values" in v:
            params[k.replace("-", "_")] = v["values"][0]  # pick first value
    return params


def make_env(env_id, video_dir, policy_name):
    dataset = minari.load_dataset(env_id)
    env = dataset.recover_environment(render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: True,
        name_prefix=f"{policy_name}-bc-eval"
    )

    return env, dataset


def load_policy(checkpoint, dataset, hidden_dim, depth, policy_name):
    obs_dim = dataset.observation_space.shape[0]
    act_dim = dataset.action_space.shape[0]

    match(policy_name):
        case "gaussian":
            policy = GaussianPolicy(
                obs_dim,
                act_dim,
                hidden_sizes=[hidden_dim] * depth
            ).to(DEVICE)
        case "flow-matching":
            policy = FlowPolicy()  # TODO: implement
        case _:
            raise ValueError(f"Unknown policy: {policy}")

    policy.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    policy.eval()

    return policy


def compute_state_stats(dataset):
    states = []

    for ep in dataset.iterate_episodes():
        states.append(torch.tensor(ep.observations[:-1], dtype=torch.float32))

    states = torch.cat(states, dim=0)
    mean = states.mean(0).to(DEVICE)
    std = states.std(0).to(DEVICE)

    return mean, std


@torch.no_grad()
def run_eval(policy, env, state_mean, state_std, episodes):
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=DEVICE)

            state = (state - state_mean) / (state_std + 1e-8)
            state = torch.clamp(state, -10.0, 10.0)
            state = state.unsqueeze(0)

            action = policy.sample(state, deterministic=True)[0]
            action = action.cpu().numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        returns.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.2f}")

    print("\nAverage return:", np.mean(returns))


def main():
    args = parse_args()
    config_params = load_config(args.config)

    env_id = config_params.get("env_name", "mujoco/halfcheetah/medium-v0")
    checkpoint = args.checkpoint or config_params.get("save_path")
    episodes = args.num_episodes or config_params.get("eval_episodes", 3)
    hidden_dim = config_params.get("hidden_dim", 256)
    depth = config_params.get("depth", 2)
    policy_name = config_params.get("policy", "gaussian")

    video_dir = os.path.join(args.video_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(video_dir, exist_ok=True)

    env, dataset = make_env(env_id, video_dir, policy_name)
    policy = load_policy(checkpoint, dataset, hidden_dim, depth, policy_name)
    state_mean, state_std = compute_state_stats(dataset)

    run_eval(policy, env, state_mean, state_std, episodes)
    env.close()


if __name__ == "__main__":
    main()