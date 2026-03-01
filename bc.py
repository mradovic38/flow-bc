import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb


class BehavioralCloning:
    def __init__(
        self,
        policy: nn.Module,
        expert_dataset,
        device="cpu",
        batch_size=128,
        num_epochs=1000,
        lr=1e-3,
        state_norm=True,
        action_in_norm=False, # tanh squashed shouldn't normalize
        action_out_denorm=False, # tanh squashed shouldn't normalize
        bc_noise=None,
        max_grad_norm=2.0,
        eval_env=None,
        eval_interval=50,
        eval_episodes=3,
        save_path=None,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr = lr

        self.state_norm = state_norm
        self.action_in_norm = action_in_norm
        self.action_out_denorm = action_out_denorm
        self.bc_noise = bc_noise
        self.max_grad_norm = max_grad_norm

        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_path = save_path

        # Expert dataset and dataloader
        self.expert_dataset = expert_dataset
        self.dataloader = DataLoader(
            expert_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        # Precompute normalization stats
        states = torch.stack([d["state"] for d in expert_dataset], dim=0)
        actions = torch.stack([d["actions"] for d in expert_dataset], dim=0)
        self.state_mean, self.state_std = states.mean(0), states.std(0)
        self.action_mean, self.action_std = actions.mean(0), actions.std(0)
        self.state_mean = self.state_mean.to(self.device)
        self.state_std = self.state_std.to(self.device)
        self.action_mean = self.action_mean.to(self.device)
        self.action_std = self.action_std.to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if wandb.run is not None:  # assume main script already did wandb.init()
            self.use_wandb = True
            wandb.watch(self.policy, log="all", log_freq=100)
            # wandb.config.update({
            #     "batch_size": batch_size,
            #     "lr": lr,
            #     "num_epochs": num_epochs,
            #     "state_norm": state_norm,
            #     "bc_noise": bc_noise,
            #     "grad_clip": max_grad_norm
            # }, allow_val_change=True)
        else:
            self.use_wandb = False

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_norm:
            state = (state - self.state_mean) / (self.state_std + 1e-8)
            state = torch.clamp(state, -10.0, 10.0)
        return state

    def _normalize_action_in(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_in_norm:
            action = (action - self.action_mean.to(self.device)) / (self.action_std.to(self.device) + 1e-8)
        return action

    def _denorm_action_out(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_out_denorm:
            action = action * (self.action_std.to(self.device) + 1e-8) + self.action_mean.to(self.device)
        return action

    def _get_batch(self, batch):
        states = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device)
        states = self._normalize_state(states)
        actions = self._normalize_action_in(actions)
        if self.bc_noise is not None:
            states += torch.randn_like(states) * self.bc_noise
        return states, actions

    def train(self):
        self.policy.train()
        best_return = -np.inf

        epoch_bar = tqdm(range(self.num_epochs), desc="BC Training", leave=True)

        for epoch in epoch_bar:
            total_loss = 0.0
            batch_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                leave=False
            )

            for batch_idx, batch in enumerate(batch_bar):
                states, actions = self._get_batch(batch)
                loss = -self.policy.log_prob(states, actions).mean()

                self.optimizer.zero_grad()
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                batch_bar.set_postfix(loss=loss.item())

                if self.use_wandb:
                    global_step = epoch * len(self.dataloader) + batch_idx
                    wandb.log({
                        "train/grad_norm": grad_norm.item(),
                        "train/loss": loss.item()
                    })

            avg_loss = total_loss / len(self.dataloader)
            log_dict = {"avg_loss": avg_loss}

            avg_return = None
            if self.eval_env is not None and (epoch + 1) % self.eval_interval == 0:
                avg_return = self.evaluate()
                log_dict["return"] = avg_return
                epoch_bar.write(f"Epoch {epoch+1} | Avg Loss {avg_loss:.6f} | Avg Return {avg_return:.2f}")

                if avg_return > best_return and self.save_path is not None:
                    best_return = avg_return
                    torch.save(self.policy.state_dict(), self.save_path)

            if self.use_wandb:
                wandb.log({
                    "train/avg_loss": avg_loss,
                    "eval/return": avg_return if avg_return is not None else None
                })

            epoch_bar.set_postfix(**log_dict)

    def evaluate(self):
        if self.eval_env is None:
            return None

        self.policy.eval()
        returns = []

        # pbar over eval episodes
        pbar = tqdm(range(self.eval_episodes), desc="Evaluating", leave=False)
        with torch.no_grad():
            for _ in pbar:
                obs, _ = self.eval_env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    action = self.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                    done = terminated or truncated
                    total_reward += reward

                returns.append(total_reward)
                pbar.set_postfix({"return": total_reward})

        self.policy.train()
        return float(np.mean(returns))

    def predict(self, state, deterministic=False):
        self.policy.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            )

            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            state_tensor = self._normalize_state(state_tensor)

            action = self.policy.sample(state_tensor, deterministic=deterministic)
            action = self._denorm_action_out(action)

            if action.shape[0] == 1:
                action = action[0]

            return action.cpu().numpy()