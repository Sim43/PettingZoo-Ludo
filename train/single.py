import os
import time
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ludo.ludo import env as ludo_env


# ==============================
# Config
# ==============================
NUM_ENVS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "train/checkpoints/single.pt"
BEST_CHECKPOINT_PATH = "train/checkpoints/single_best.pt"

# PPO hyperparams
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
LR_MIN = 1e-5  # Minimum learning rate for cosine annealing
CLIP_EPS = 0.2
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_END = 0.01
ENTROPY_DECAY_EPISODES = 50000
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

# Rollout / update
ROLLOUT_EPISODES_PER_ENV = 8   # collect this many full games per env before updating
PPO_EPOCHS = 4                 # gradient passes over collected data
MINIBATCH_SIZE = 2048          # transitions per minibatch (auto-clipped to data size)

# Training duration
MAX_EPISODES = 100000  # Stop training after this many episodes
EVAL_INTERVAL = 500    # Evaluate every N episodes
EVAL_EPISODES = 20     # Number of episodes for evaluation

# Expected training time estimates (approximate)
# Based on: ~2-5 seconds per game, 5 envs * 8 games = 40 games per update
# ~80-200 seconds per update, ~100k episodes = ~2500 updates = ~55-140 hours
ESTIMATED_HOURS_FOR_GOOD_RESULTS = 24  # Conservative estimate for decent performance
ESTIMATED_HOURS_FOR_EXCELLENT = 72     # Estimate for strong performance


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
    reward: float              # Step reward (not cumulative)
    done: bool                 # done for that agent at that time step


def run_episode_collect(env, model: ActorCritic, render=False) -> Dict[str, List[Transition]]:
    """
    Runs ONE full PettingZoo AEC game.
    Collects per-agent transitions (self-play with shared weights).
    
    CRITICAL: In PettingZoo AEC, env.last() returns CUMULATIVE reward since agent last acted.
    We need to compute step rewards by taking the difference between consecutive calls.
    This ensures we capture:
    - Step rewards (shaping rewards, captures, etc.)
    - Terminal rewards (final game ending rewards)
    """
    env.reset()
    per_agent: Dict[str, List[Transition]] = {a: [] for a in env.agents}
    
    # Track cumulative rewards for each agent to compute step rewards
    prev_cumulative_rewards: Dict[str, float] = {a: 0.0 for a in env.agents}

    while env.agents:
        agent = env.agent_selection
        obs, reward_cumulative, term, trunc, info = env.last()

        done = bool(term or trunc)

        if done:
            # Agent is done - capture any final reward before stepping
            step_reward = reward_cumulative - prev_cumulative_rewards.get(agent, 0.0)
            if agent in per_agent and len(per_agent[agent]) > 0:
                # Add final reward to last transition if agent had transitions
                per_agent[agent][-1].reward += step_reward
            prev_cumulative_rewards[agent] = reward_cumulative
            env.step(None)
            continue

        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,obs]
        mask_t = torch.tensor(info["action_mask"], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1,act]

        with torch.no_grad():
            logits, value = model(obs_t)
            dist = masked_categorical(logits, mask_t)
            action = dist.sample()
            logprob = dist.log_prob(action)

        # Get reward before step (cumulative up to now)
        prev_reward = prev_cumulative_rewards.get(agent, 0.0)
        
        # Take action
        env.step(action.item())
        
        # Get reward after step (new cumulative reward)
        obs_next, reward_next, term_next, trunc_next, info_next = env.last()
        new_cumulative_reward = float(reward_next)
        
        # Compute step reward: difference between cumulative rewards
        step_reward = new_cumulative_reward - prev_reward
        prev_cumulative_rewards[agent] = new_cumulative_reward

        per_agent[agent].append(
            Transition(
                obs=obs_t.squeeze(0).detach(),
                mask=mask_t.squeeze(0).detach(),
                action=action.squeeze(0).detach(),
                logprob=logprob.squeeze(0).detach(),
                value=value.squeeze(0).detach(),
                reward=step_reward,  # Store step reward, not cumulative
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
        next_value = values[t]  # This becomes the bootstrap for t-1

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
    
    # Normalize advantages across entire batch (better than per-trajectory)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return obs, mask, actions, old_logp, returns, advantages, old_values


def ppo_update(model: ActorCritic, optimizer, batch, epochs: int, minibatch_size: int, entropy_coef: float):
    obs, mask, actions, old_logp, returns, advantages, old_values = batch
    N = obs.shape[0]
    minibatch_size = min(minibatch_size, N)
    
    # Ensure minibatch_size divides N to avoid wasting data
    if N % minibatch_size != 0:
        minibatch_size = N // (N // minibatch_size)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_grad_norm = 0.0

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

            loss = policy_loss + VALUE_COEF * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_grad_norm += grad_norm

    num_updates = epochs * (N // minibatch_size)
    return {
        'policy_loss': total_policy_loss / num_updates,
        'value_loss': total_value_loss / num_updates,
        'entropy': total_entropy / num_updates,
        'grad_norm': total_grad_norm / num_updates,
    }


# ==============================
# Evaluation
# ==============================
def evaluate(model: ActorCritic, num_episodes: int = EVAL_EPISODES) -> Dict[str, float]:
    """Run evaluation episodes and return statistics."""
    eval_env = ludo_env(render_mode=None)
    
    # Track statistics
    episode_lengths = []
    returns_per_agent: Dict[str, List[float]] = {}
    wins_per_agent: Dict[str, int] = {}
    total_episodes = 0
    
    for _ in range(num_episodes):
        data = run_episode_collect(eval_env, model, render=False)
        
        # Track episode length (sum of all agent steps)
        episode_length = sum(len(traj) for traj in data.values())
        episode_lengths.append(episode_length)
        
        # Track returns and wins
        for agent, traj in data.items():
            if agent not in returns_per_agent:
                returns_per_agent[agent] = []
                wins_per_agent[agent] = 0
            
            if traj:
                # Sum of rewards for this agent
                agent_return = sum(t.reward for t in traj)
                returns_per_agent[agent].append(agent_return)
                
                # Check if agent won (highest return in this episode)
                episode_returns = {a: sum(t.reward for t in traj) for a, traj in data.items() if traj}
                if episode_returns:
                    max_return = max(episode_returns.values())
                    if agent_return == max_return and agent_return > 0:
                        wins_per_agent[agent] += 1
        
        total_episodes += 1
    
    eval_env.close()
    
    # Compute statistics
    stats = {
        'avg_episode_length': np.mean(episode_lengths) if episode_lengths else 0.0,
        'avg_return': np.mean([np.mean(returns_per_agent[a]) for a in returns_per_agent]) if returns_per_agent else 0.0,
        'win_rate': np.mean([wins_per_agent[a] / total_episodes for a in wins_per_agent]) if wins_per_agent else 0.0,
    }
    
    # Per-agent statistics
    for agent in returns_per_agent:
        stats[f'{agent}_avg_return'] = np.mean(returns_per_agent[agent])
        stats[f'{agent}_win_rate'] = wins_per_agent[agent] / total_episodes if total_episodes > 0 else 0.0
    
    return stats


# ==============================
# Main Training Loop
# ==============================
def save_checkpoint(model, optimizer, scheduler, episode_count: int, path: str = CHECKPOINT_PATH):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "episode_count": episode_count,
    }
    torch.save(payload, path)


def save_best_checkpoint(model, optimizer, scheduler, episode_count: int, win_rate: float):
    """Save checkpoint if win_rate is the best seen so far."""
    best_path = BEST_CHECKPOINT_PATH
    best_winrate_path = BEST_CHECKPOINT_PATH.replace(".pt", "_winrate.txt")
    
    best_winrate = float("-inf")
    if os.path.exists(best_winrate_path):
        with open(best_winrate_path, "r") as f:
            best_winrate = float(f.read().strip())
    
    if win_rate > best_winrate:
        save_checkpoint(model, optimizer, scheduler, episode_count, best_path)
        with open(best_winrate_path, "w") as f:
            f.write(str(win_rate))
        print(f"New best checkpoint saved! Win rate: {win_rate:.3f} (prev: {best_winrate:.3f})")
        return True
    return False


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: Optional[str] = None):
    """Load checkpoint from specified path or default CHECKPOINT_PATH."""
    path = checkpoint_path or CHECKPOINT_PATH
    if not os.path.exists(path):
        return 0
    payload = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return int(payload.get("episode_count", 0))


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: Environment seeds are set per-episode in reset()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def main():
    # Set seed for reproducibility
    set_seed(42)
    
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    model = ActorCritic().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPISODES, eta_min=LR_MIN)

    episode = load_checkpoint(model, optimizer, scheduler)
    if episode > 0:
        print(f"Loaded checkpoint at episode {episode}.")
    
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Environments: {NUM_ENVS}")
    print(f"  Rollout episodes per env: {ROLLOUT_EPISODES_PER_ENV}")
    print(f"  Max episodes: {MAX_EPISODES}")
    print(f"  Expected time for good results: ~{ESTIMATED_HOURS_FOR_GOOD_RESULTS} hours")
    print(f"  Expected time for excellent results: ~{ESTIMATED_HOURS_FOR_EXCELLENT} hours")
    print(f"{'='*60}\n")

    envs = [ludo_env(render_mode=None) for _ in range(NUM_ENVS)]
    
    start_time = time.time()
    last_eval_episode = episode
    best_win_rate = float("-inf")

    try:
        while episode < MAX_EPISODES:
            # -------- collect rollouts --------
            collect_start = time.time()
            per_env_data = []
            games_this_update = NUM_ENVS * ROLLOUT_EPISODES_PER_ENV

            for env in envs:
                for _ in range(ROLLOUT_EPISODES_PER_ENV):
                    per_env_data.append(run_episode_collect(env, model, render=False))

            episode += games_this_update
            collect_time = time.time() - collect_start

            batch = flatten_batch(per_env_data)
            if batch is None:
                # extremely unlikely, but keep training loop robust
                continue

            # -------- PPO update --------
            update_start = time.time()
            # Entropy decay: start at ENTROPY_COEF_START, decay to ENTROPY_COEF_END
            current_entropy = max(
                ENTROPY_COEF_END,
                ENTROPY_COEF_START * (1.0 - episode / ENTROPY_DECAY_EPISODES)
            )
            update_stats = ppo_update(
                model, optimizer, batch, epochs=PPO_EPOCHS,
                minibatch_size=MINIBATCH_SIZE, entropy_coef=current_entropy
            )
            scheduler.step()
            update_time = time.time() - update_start

            # -------- checkpoint overwrite --------
            save_checkpoint(model, optimizer, scheduler, episode)

            # -------- evaluation --------
            eval_stats = None
            if episode - last_eval_episode >= EVAL_INTERVAL:
                eval_start = time.time()
                eval_stats = evaluate(model, num_episodes=EVAL_EPISODES)
                eval_time = time.time() - eval_start
                last_eval_episode = episode
                
                # Save best checkpoint based on win rate
                win_rate = eval_stats.get('win_rate', 0.0)
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    save_best_checkpoint(model, optimizer, scheduler, episode, win_rate)

            # -------- logging --------
            obs, mask, actions, old_logp, returns, advantages, old_values = batch
            avg_return = returns.mean().item()
            current_lr = scheduler.get_last_lr()[0]
            
            elapsed_time = time.time() - start_time
            episodes_per_hour = episode / (elapsed_time / 3600) if elapsed_time > 0 else 0
            estimated_time_remaining = (MAX_EPISODES - episode) / episodes_per_hour if episodes_per_hour > 0 else 0
            
            if episode % 100 == 0 or eval_stats is not None:
                print(f"\nEpisode {episode}/{MAX_EPISODES}")
                print(f"  Batch: {obs.shape[0]} transitions | Avg return: {avg_return:.3f}")
                print(f"  Policy loss: {update_stats['policy_loss']:.4f} | "
                      f"Value loss: {update_stats['value_loss']:.4f} | "
                      f"Entropy: {update_stats['entropy']:.4f}")
                print(f"  Grad norm: {update_stats['grad_norm']:.4f} | "
                      f"LR: {current_lr:.6f} | Entropy coef: {current_entropy:.4f}")
                print(f"  Time: Collect={collect_time:.1f}s | Update={update_time:.1f}s")
                print(f"  Progress: {episode/MAX_EPISODES*100:.1f}% | "
                      f"Speed: {episodes_per_hour:.1f} episodes/hour")
                print(f"  ETA: {format_time(estimated_time_remaining)}")
                
                if eval_stats:
                    print(f"\n  Evaluation (last {EVAL_EPISODES} episodes):")
                    print(f"    Win rate: {eval_stats['win_rate']:.3f} | "
                          f"Avg episode length: {eval_stats['avg_episode_length']:.1f}")
                    print(f"    Avg return: {eval_stats['avg_return']:.3f}")
                    for agent in ['player_0', 'player_1', 'player_2', 'player_3']:
                        if f'{agent}_win_rate' in eval_stats:
                            print(f"    {agent}: return={eval_stats[f'{agent}_avg_return']:.3f}, "
                                  f"win_rate={eval_stats[f'{agent}_win_rate']:.3f}")
                    print(f"  Eval time: {eval_time:.1f}s")
                print()

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at episode {episode}")
        print("Saving final checkpoint...")
        save_checkpoint(model, optimizer, scheduler, episode)
        print("Checkpoint saved.")
    
    finally:
        for env in envs:
            env.close()
        print(f"\nTraining completed. Total episodes: {episode}")
        print(f"Total time: {format_time(time.time() - start_time)}")
        print(f"Best win rate achieved: {best_win_rate:.3f}")


if __name__ == "__main__":
    main()
