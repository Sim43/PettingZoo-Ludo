import os
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from ludo.ludo import env as ludo_env


# ==============================
# Config
# ==============================
NUM_ENVS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "train/checkpoints/single.pt"
RENDER_EVERY = 1000  # episodes

# PPO hyperparams
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Rollout / update
ROLLOUT_EPISODES_PER_ENV = 2   # collect this many full games per env before updating
PPO_EPOCHS = 4                 # gradient passes over collected data
MINIBATCH_SIZE = 2048          # transitions per minibatch (auto-clipped to data size)


# ==============================
# Actor-Critic Network
# ==============================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=None, act_dim=None):
        super().__init__()
        if obs_dim is None or act_dim is None:
            # Infer observation and action dimensions from a temporary environment
            # instance so that changes to the Ludo environment are picked up
            # automatically.
            tmp_env = ludo_env(render_mode=None)
            if obs_dim is None:
                obs_dim = tmp_env.observation_space("player_0").shape[0]
            if act_dim is None:
                act_dim = tmp_env.action_space("player_0").n
            tmp_env.close()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, act_dim)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)  # [B]
        return logits, value


# ==============================
# Utilities
# ==============================
def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> torch.distributions.Categorical:
    """
    logits: [B, A]
    mask:   [B, A] float/bool where 1 = legal, 0 = illegal
    """
    masked_logits = logits.masked_fill(mask == 0, -1e9)
    probs = torch.softmax(masked_logits, dim=-1)
    return torch.distributions.Categorical(probs=probs)


@dataclass
class Transition:
    obs: torch.Tensor          # [obs_dim]
    mask: torch.Tensor         # [act_dim]
    action: torch.Tensor       # []
    logprob: torch.Tensor      # []
    value: torch.Tensor        # []
    reward: float
    done: bool                 # done for that agent at that time step


def run_episode_collect(env, model: ActorCritic, render=False) -> Dict[str, List[Transition]]:
    """
    Runs ONE full PettingZoo AEC game.
    Collects per-agent transitions (self-play with shared weights).
    Note: In AEC, agents may receive rewards on other agents' moves.
    We'll store reward seen at each of agent's own decision points (via env.last()).
    """
    env.reset()
    per_agent: Dict[str, List[Transition]] = {a: [] for a in env.agents}

    while env.agents:
        agent = env.agent_selection
        obs, reward, term, trunc, info = env.last()

        done = bool(term or trunc)

        if done:
            env.step(None)
            continue

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,obs]
        mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,act]

        with torch.no_grad():
            logits, value = model(obs_t)
            dist = masked_categorical(logits, mask_t)
            action = dist.sample()
            logprob = dist.log_prob(action)

        env.step(action.item())

        per_agent[agent].append(
            Transition(
                obs=obs_t.squeeze(0).detach(),
                mask=mask_t.squeeze(0).detach(),
                action=action.squeeze(0).detach(),
                logprob=logprob.squeeze(0).detach(),
                value=value.squeeze(0).detach(),
                reward=float(reward),
                done=False,  # agent made a decision; termination handled by env loop
            )
        )

        if render:
            env.render()
            time.sleep(0.75)

    return per_agent


def compute_gae_and_returns(traj: List[Transition], gamma: float, lam: float):
    """
    Compute advantages (GAE-Lambda) and returns for one agent's trajectory.
    We treat terminal bootstrap value as 0 because the episode ends.
    """
    if not traj:
        return None, None

    rewards = torch.tensor([t.reward for t in traj], dtype=torch.float32, device=DEVICE)
    values = torch.stack([t.value for t in traj]).to(torch.float32)  # [T]

    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32, device=DEVICE)

    gae = 0.0
    next_value = 0.0  # terminal bootstrap

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def flatten_batch(per_env_data: List[Dict[str, List[Transition]]]):
    """
    Flatten list of per-env dicts into a single transition batch.
    Returns tensors:
      obs [N,obs], mask [N,act], actions [N], old_logp [N], returns [N], adv [N], old_values [N]
    """
    obs_list, mask_list, act_list, oldlp_list, ret_list, adv_list, val_list = [], [], [], [], [], [], []

    for env_dict in per_env_data:
        for _, traj in env_dict.items():
            if not traj:
                continue
            adv, ret = compute_gae_and_returns(traj, GAMMA, GAE_LAMBDA)
            if adv is None:
                continue

            # normalize advantages per-trajectory (simple + stable)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs_list.append(torch.stack([t.obs for t in traj]))
            mask_list.append(torch.stack([t.mask for t in traj]))
            act_list.append(torch.stack([t.action for t in traj]).long())
            oldlp_list.append(torch.stack([t.logprob for t in traj]))
            val_list.append(torch.stack([t.value for t in traj]))
            ret_list.append(ret)
            adv_list.append(adv)

    if not obs_list:
        return None

    obs = torch.cat(obs_list, dim=0)
    mask = torch.cat(mask_list, dim=0)
    actions = torch.cat(act_list, dim=0)
    old_logp = torch.cat(oldlp_list, dim=0)
    old_values = torch.cat(val_list, dim=0)
    returns = torch.cat(ret_list, dim=0)
    advantages = torch.cat(adv_list, dim=0)

    return obs, mask, actions, old_logp, returns, advantages, old_values


def ppo_update(model: ActorCritic, optimizer, batch, epochs: int, minibatch_size: int):
    obs, mask, actions, old_logp, returns, advantages, old_values = batch
    N = obs.shape[0]
    minibatch_size = min(minibatch_size, N)

    for _ in range(epochs):
        idx = torch.randperm(N, device=DEVICE)
        for start in range(0, N, minibatch_size):
            mb_idx = idx[start:start + minibatch_size]

            mb_obs = obs[mb_idx]
            mb_mask = mask[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logp = old_logp[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            logits, values = model(mb_obs)
            dist = masked_categorical(logits, mb_mask)

            new_logp = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (mb_returns - values).pow(2).mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()


# ==============================
# Main Training Loop
# ==============================
def save_checkpoint(model, optimizer, episode_count: int):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode_count": episode_count,
    }
    torch.save(payload, CHECKPOINT_PATH)


def load_checkpoint(model, optimizer):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0
    payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("episode_count", 0))


def main():
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    episode = load_checkpoint(model, optimizer)
    if episode > 0:
        print(f"Loaded checkpoint at episode {episode}.")

    envs = [ludo_env(render_mode=None) for _ in range(NUM_ENVS)]

    while True:
        # -------- collect rollouts --------
        per_env_data = []
        games_this_update = NUM_ENVS * ROLLOUT_EPISODES_PER_ENV

        for env in envs:
            for _ in range(ROLLOUT_EPISODES_PER_ENV):
                per_env_data.append(run_episode_collect(env, model, render=False))

        episode += games_this_update

        batch = flatten_batch(per_env_data)
        if batch is None:
            # extremely unlikely, but keep training loop robust
            continue

        # -------- PPO update --------
        ppo_update(model, optimizer, batch, epochs=PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE)

        # -------- checkpoint overwrite --------
        save_checkpoint(model, optimizer, episode)

        if episode % 100 == 0:
            obs, mask, actions, old_logp, returns, advantages, old_values = batch
            avg_return = returns.mean().item()
            print(f"Episode {episode} | Batch transitions {obs.shape[0]} | Avg return {avg_return:.3f}")
            

if __name__ == "__main__":
    main()
