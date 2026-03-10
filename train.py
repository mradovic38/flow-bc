import time
from dotenv import load_dotenv
import argparse
import yaml
import wandb

def flatten_trajectory_obs(obs_obj):
    """
    Recursively flattens a nested dictionary of numpy arrays (spanning time T)
    into a single list of 2D arrays (T, flattened_dim).
    """
    if isinstance(obs_obj, dict):
        arrays = []
        # Gymnasium flattens dict spaces by sorting keys alphabetically
        for k in sorted(obs_obj.keys()):
            arrays.extend(flatten_trajectory_obs(obs_obj[k]))
        return arrays
    else:
        # We've reached the actual numpy array of shape (T, ...)
        return [obs_obj.reshape(obs_obj.shape[0], -1)]
    
def main(config):
    import torch, random, numpy as np
    import minari
    import gymnasium as gym
    from policy import GaussianPolicy, FlowMatchingPolicy
    from bc import BehavioralCloning

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Load dataset
    env_name = config.get("env_name", "D4RL/kitchen/mixed-v2") # <-- Default updated
    dataset = minari.load_dataset(env_name)
    
    eval_env = dataset.recover_environment(eval_env=True)

    is_dict_space = isinstance(eval_env.observation_space, gym.spaces.Dict)
    if is_dict_space:
        eval_env = gym.wrappers.FlattenObservation(eval_env)

    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]

    # Build policy
    hidden_dim = config.get("hidden_dim", 256)
    depth = config.get("depth", 2)
    policy_type = config.get("policy", "gaussian")
    hidden_sizes = [hidden_dim] * depth

    match policy_type:
        case "gaussian":
            policy = GaussianPolicy(
                obs_dim=obs_dim, 
                act_dim=action_dim, 
                hidden_sizes=hidden_sizes
            ).to(device)

        case "flow-matching":
            time_freq_dim = config.get("time_freq_dim", 64)
            ode_steps = config.get("ode_steps", 15)
            ode_method = config.get("ode_method", "euler")
            velocity_hidden_sizes = [config.get("velocity_hidden_dim", 256)] * config.get("velocity_depth", 2)
            time_embedder_hidden_size = config.get("time_embedder_hidden_dim", 256)
            ema_decay = config.get("ema_decay", 0.9999)
            lognormal_mu = config.get("lognormal_mu", -1.2)
            lognormal_sigma = config.get("lognormal_sigma", 1.2)

            policy = FlowMatchingPolicy(
                obs_dim=obs_dim, 
                act_dim=action_dim, 
                backbone_hidden_sizes=hidden_sizes, 
                velocity_hidden_sizes=velocity_hidden_sizes, 
                time_embedder_hidden_size=time_embedder_hidden_size, 
                time_freq_dim=time_freq_dim, 
                ode_steps=ode_steps, 
                ode_method=ode_method, 
                ema_decay=ema_decay, 
                lognormal_mu=lognormal_mu, 
                lognormal_sigma=lognormal_sigma
            ).to(device)
        case _:
            raise ValueError(f"Unknown policy: {policy_type}")

    expert_data = []

    for ep_i in range(dataset.total_episodes):
        episode = dataset[ep_i]

        if is_dict_space:
            obs_arrays = flatten_trajectory_obs(episode.observations)
            obs = np.concatenate(obs_arrays, axis=-1)
        else:
            obs = episode.observations

        acts = episode.actions

        T = min(len(obs) - 1, len(acts))

        states = torch.tensor(obs[:-1], dtype=torch.float32)
        actions = torch.tensor(acts[:T], dtype=torch.float32)

        for s, a in zip(states, actions):
            expert_data.append({"state": s, "actions": a})

    bc_trainer = BehavioralCloning(
        policy=policy,
        expert_dataset=expert_data,
        device=device,
        batch_size=config.get("batch_size", 256),
        num_epochs=config.get("num_epochs", 50),
        lr=config.get("lr", 3e-4),
        eval_env=eval_env,
        eval_interval=config.get("eval_interval", 10),
        eval_episodes=config.get("eval_episodes", 3),
        save_path=config.get("save_path", f"{env_name.replace('/', '_')}.pt"),
    )

    print("Starting BC training...")
    bc_trainer.train()
    print("Finished training.")


def parse_unknown_args(unknown_args):
    """
    Convert unknown CLI args like --batch-size=512 to a dict
    """
    cfg = {}
    for arg in unknown_args:
        if arg.startswith("--") and "=" in arg:
            key, val = arg.lstrip("-").split("=", 1)
            key = key.replace("-", "_")
            try:
                val = eval(val)  # convert to int, float, bool, list if possible
            except:
                pass
            cfg[key] = val
    return cfg


if __name__ == "__main__":
    start = time.time()
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")
    args, unknown_args = parser.parse_known_args()
    cfg_dict = {}

    project_name = None
    run_name = None
    if args.config:
        with open(args.config) as f:
            loaded_yaml = yaml.safe_load(f)

        # Extract project and base run name
        project_name = loaded_yaml.get("project")
        run_name = loaded_yaml.get("name")

        # Flatten parameters for single run
        for k, v in loaded_yaml.get("parameters", {}).items():
            key_clean = k.replace("-", "_")
            if "value" in v:
                cfg_dict[key_clean] = v["value"]
            elif "values" in v:
                cfg_dict[key_clean] = v["values"][0]  # pick first value for single run

    # Merge unknown CLI args (from sweep agent)
    unknown_cfg = parse_unknown_args(unknown_args)
    cfg_dict.update(unknown_cfg)
    wandb_mode = "disabled" if args.disable_wandb else "online"

    base_name = run_name or "run"
    params_list = [f"{k}:{v}" for k, v in cfg_dict.items() if v is not None]
    run_final_name = f"{base_name}-{'-'.join(params_list)}" if params_list else base_name

    wandb_kwargs = {
        "config": cfg_dict,
        "mode": wandb_mode,
    }
    if project_name:
        wandb_kwargs["project"] = project_name
    wandb_kwargs["name"] = run_final_name
    run = wandb.init(**wandb_kwargs)
    config = wandb.config

    main(config)

    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")
