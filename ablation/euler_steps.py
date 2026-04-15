import os
import argparse
import datetime
import yaml
import random
import time

import numpy as np
import torch
import minari
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from policy import GaussianPolicy, FlowMatchingPolicy

# Force PyTorch to use a single thread for rigorous CPU benchmarking
torch.set_num_threads(1)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Flow Matching model across multiple ODE steps")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint override")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory to save videos")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of episodes to run per ODE step count")
    
    parser.add_argument(
        "--ode-steps-list", 
        nargs="+", 
        type=int, 
        default=[1, 2, 4, 8, 10, 12, 16, 20, 24, 36, 50], 
        help="List of ODE steps to evaluate (e.g., --ode-steps-list 1 5 10 20)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed to evaluate (e.g., --seed 42)"
    )

    return parser.parse_args()


def set_seed(seed):
    """Sets global seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def make_env(env_id, video_dir, policy_name, step_suffix=""):
    dataset = minari.load_dataset(env_id)
    env = dataset.recover_environment(render_mode="rgb_array")

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)

    if video_dir is not None:
        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda ep: True,
            name_prefix=f"{policy_name}-bc-eval{step_suffix}"
        )

    return env, dataset


def load_policy(checkpoint, env, dataset, policy_name, config, ode_steps=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    hidden_dim = config.get("hidden_dim", 256)
    depth = config.get("depth", 2)

    hidden_sizes = [hidden_dim] * depth

    match policy_name:
        case "gaussian":
            policy = GaussianPolicy(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_sizes=hidden_sizes
            ).to(DEVICE)
        case "flow-matching":
            time_freq_dim = config.get("time_freq_dim", 64)
            ode_method = config.get("ode_method", "euler")
            velocity_hidden_sizes = [config.get("velocity_hidden_dim", 256)] * config.get("velocity_depth", 2)
            time_embedder_hidden_size = config.get("time_embedder_hidden_dim", 256)
            ema_decay = config.get("ema_decay", 0.9999)
            lognormal_mu = config.get("lognormal_mu", -1.2)
            lognormal_sigma = config.get("lognormal_sigma", 1.2)

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
            raise ValueError(f"Unknown policy: {policy_name}")

    checkpoint_data = torch.load(checkpoint, map_location=DEVICE)
    if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
        policy.load_state_dict(checkpoint_data["model"])

        if hasattr(policy, "ema") and "ema_shadow" in checkpoint_data:
            policy.ema.shadow = {k: v.to(DEVICE) for k, v in checkpoint_data["ema_shadow"].items()}

        state_mean = checkpoint_data["state_mean"].to(DEVICE)
        state_std = checkpoint_data["state_std"].to(DEVICE)
    else:
        state_mean, state_std = compute_state_stats(dataset, env)
        policy.load_state_dict(checkpoint_data)

    policy.eval()
    if hasattr(policy, "ema"):
        policy.ema.apply_shadow()

    return policy, state_mean, state_std


def flatten_trajectory_obs(obs_obj):
    """Recursively flattens nested dictionaries (same as training script)"""
    if isinstance(obs_obj, dict):
        arrays = []
        for k in sorted(obs_obj.keys()):
            arrays.extend(flatten_trajectory_obs(obs_obj[k]))
        return arrays
    else:
        return [obs_obj.reshape(obs_obj.shape[0], -1)]


def compute_state_stats(dataset, env):
    states = []
    is_dict_space = isinstance(env.observation_space, gym.spaces.Dict)

    for ep in dataset.iterate_episodes():
        if is_dict_space:
            obs_arrays = flatten_trajectory_obs(ep.observations)
            obs = np.concatenate(obs_arrays, axis=-1)
        else:
            obs = ep.observations
            
        states.append(torch.tensor(obs[:-1], dtype=torch.float32))

    states = torch.cat(states, dim=0)
    mean = states.mean(0).to(DEVICE)
    std = states.std(0).to(DEVICE)

    return mean, std


@torch.no_grad()
def benchmark_neutral_latency(policy, obs_dim, num_warmup=20, num_iters=500):
    """Runs a sterile, isolated benchmark for inference latency."""
    policy_device = next(policy.parameters()).device
    dummy_obs = torch.randn(1, obs_dim, device=policy_device)

    for _ in range(num_warmup):
        _ = policy.sample(dummy_obs, deterministic=True)

    latencies = []
    
    for _ in range(num_iters):
        start_time = time.perf_counter()
        _ = policy.sample(dummy_obs, deterministic=True)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

    return np.mean(latencies)


@torch.no_grad()
def run_eval(policy, env, state_mean, state_std, episodes, base_seed):
    """Evaluates the policy strictly for environmental returns."""
    set_seed(base_seed)
    returns = []

    for ep in range(episodes):
        env_seed = (base_seed * 10000) + ep
        obs, _ = env.reset(seed=env_seed)
        
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
        print(f"  Episode {ep+1}: {total_reward:.2f}")

    return returns


def main():
    args = parse_args()
    config_params = load_config(args.config)

    env_id = config_params.get("env_name", "mujoco/pusher/medium-v0")
    checkpoint = args.checkpoint or config_params.get("save_path")
    episodes = args.num_episodes or config_params.get("eval_episodes", 3)
    policy_name = config_params.get("policy", "flow-matching")
    
    ode_steps_list = args.ode_steps_list
    seed = args.seed

    print("Config Parameters:", config_params)
    print(f"Evaluating with seed: {seed}")
    print(f"ODE Steps to evaluate: {ode_steps_list}")

    dataset = minari.load_dataset(env_id)
    
    temp_env = dataset.recover_environment()
    if isinstance(temp_env.observation_space, gym.spaces.Dict):
        temp_env = gym.wrappers.FlattenObservation(temp_env)
        
    obs_dim = temp_env.observation_space.shape[0]

    policy, state_mean, state_std = load_policy(
        checkpoint, temp_env, dataset, policy_name, config_params, ode_steps_list[0]
    )
    temp_env.close()

    results = {}

    for steps in ode_steps_list:
        print(f"\n--- Evaluating Seed {seed} with {steps} ODE steps ---")
        
        policy.ode_steps = steps

        video_dir = None
        if args.video_dir is not None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = os.path.join(args.video_dir, f"{policy_name}_ode{steps}_{timestamp}")
            os.makedirs(video_dir, exist_ok=True)

        env, _ = make_env(env_id, video_dir, policy_name, step_suffix=f"-ode{steps}")

        episode_returns = run_eval(policy, env, state_mean, state_std, episodes, seed)
        env.close()
        
        latency_ms = benchmark_neutral_latency(policy, obs_dim)

        mean_return = np.mean(episode_returns)
        std_return = np.std(episode_returns, ddof=1) if len(episode_returns) > 1 else 0.0
        
        results[steps] = {
            "mean": mean_return,
            "std": std_return,
            "latency": latency_ms
        }

    print("\n======================================================================")
    print(f"Summary Results for Seed {seed} ({episodes} episodes each)")
    print("======================================================================")
    print(f"{'ODE Steps':<12} | {'Mean Return':<15} | {'Std Return':<15} | {'Latency (ms)':<15}")
    print("-" * 70)
    for steps in ode_steps_list:
        mean = results[steps]["mean"]
        std = results[steps]["std"]
        latency = results[steps]["latency"]
        print(f"{steps:<12} | {mean:<15.2f} | {std:<15.2f} | {latency:<15.2f}")
    print("======================================================================")


if __name__ == "__main__":
    main()