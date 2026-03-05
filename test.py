import os
import argparse
import datetime
import yaml

import numpy as np
import torch
import minari
from gymnasium.wrappers import RecordVideo

from policy import GaussianPolicy, FlowMatchingPolicy

DEVICE="cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC model from YAML config with video recording")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint override")
    parser.add_argument("--video-dir", type=str, default="./videos", help="Directory to save videos")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--ode-steps", type=int, default=5, help="Number of ODE steps for flow matching policy")
    
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


def load_policy(checkpoint, dataset, policy_name, config, ode_steps=None):
    obs_dim = dataset.observation_space.shape[0]
    act_dim = dataset.action_space.shape[0]

    hidden_dim = config.get("hidden-dim", 256)
    depth = config.get("depth", 2)

    hidden_sizes=[hidden_dim]*depth

    match policy_name:
        case "gaussian":
            policy = GaussianPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim, 
                hidden_sizes=hidden_sizes
            ).to(DEVICE)
        case "flow-matching":
            time_freq_dim = config.get("time-freq-dim", 64)
            ode_method = config.get("ode-method", "euler")
            velocity_hidden_sizes = [config.get("velocity-hidden-size", 256)] * config.get("velocity-depth", 2)
            time_embedder_hidden_size = config.get("time-embedder-hidden-size", 256)
            ema_decay = config.get("ema-decay", 0.9999)
            lognormal_mu = config.get("lognormal-mu", 0.0)
            lognormal_sigma = config.get("lognormal-sigma", 0.3)
            
            policy = FlowMatchingPolicy(
                obs_dim=obs_dim, 
                act_dim=act_dim, 
                backbone_hidden_sizes=hidden_sizes, 
                velocity_hidden_sizes=velocity_hidden_sizes, 
                time_embedder_hidden_size=time_embedder_hidden_size, 
                time_freq_dim=time_freq_dim, 
                ode_steps=ode_steps, 
                ode_method=ode_method, 
                ema_decay=ema_decay, 
                lognormal_mu=lognormal_mu, 
                lognormal_sigma=lognormal_sigma
            ).to(DEVICE)
        case _:
            raise ValueError(f"Unknown policy: {policy}")

    checkpoint_data = torch.load(checkpoint, map_location=DEVICE)
    if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
        policy.load_state_dict(checkpoint_data["model"])

        if hasattr(policy, "ema") and "ema_shadow" in checkpoint_data:
            policy.ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint_data["ema_shadow"].items()}

        state_mean = checkpoint_data["state_mean"].to(DEVICE)
        state_std = checkpoint_data["state_std"].to(DEVICE)
    # TODO: remove else block once gaussian baseline is retrained / converted to ["model"] format
    else:
        state_mean, state_std = compute_state_stats(dataset)
        policy.load_state_dict(checkpoint_data)

    policy.eval()
    if hasattr(policy, "ema"):
        policy.ema.apply_shadow()

    return policy, state_mean, state_std


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
    policy_name = config_params.get("policy", "gaussian")

    ode_steps = args.ode_steps or config_params.get("ode_steps", 20)

    video_dir = os.path.join(args.video_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(video_dir, exist_ok=True)

    env, dataset = make_env(env_id, video_dir, policy_name)
    policy, state_mean, state_std = load_policy(checkpoint, dataset, policy_name, config_params, ode_steps)    

    run_eval(policy, env, state_mean, state_std, episodes)
    env.close()


if __name__ == "__main__":
    main()