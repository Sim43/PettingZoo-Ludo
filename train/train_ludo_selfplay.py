import os
import time
import argparse
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:  # pragma: no cover - import-time help
    raise SystemExit(
        "This training script requires PyTorch.\n"
        "Install it with, for example:\n"
        "  pip install torch\n"
    ) from e

from ludo.ludo import env as make_ludo_env


class PolicyNetwork(nn.Module):
    """
    Simple shared policy network for all agents.

    Input:  observation (80,) concatenated with 4-dim one-hot agent id -> (84,)
    Output: logits over 5 discrete actions.
    """

    def __init__(self, obs_dim: int = 80, num_agents: int = 4, n_actions: int = 5):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_agents = num_agents
        self.n_actions = n_actions

        input_dim = obs_dim + num_agents
        hidden = 128

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, obs_dim)
        agent_ids: (B,) integer agent indices in [0, num_agents)
        """
        # One-hot encode agent ids
        agent_oh = torch.nn.functional.one_hot(
            agent_ids.long(), num_classes=self.num_agents
        ).float()
        x = torch.cat([obs, agent_oh], dim=-1)
        return self.net(x)


def save_checkpoint(
    checkpoint_path: str,
    model: PolicyNetwork,
    optimizer: optim.Optimizer,
    episodes_done: int,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "episodes_done": episodes_done,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model: PolicyNetwork,
    optimizer: optim.Optimizer,
) -> int:
    """
    Load checkpoint if it exists.
    Returns the number of episodes already completed (0 if none).
    """
    if not os.path.exists(checkpoint_path):
        return 0

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt.get("episodes_done", 0))


def run_episode(
    env,
    model: PolicyNetwork,
    device: torch.device,
    gamma: float = 0.99,
) -> float:
    """
    Run a single self-play episode using REINFORCE with a shared policy.

    Returns:
        total_return (sum of rewards over all agents for logging).
    """

    env.reset()

    agents = env.possible_agents
    agent_to_idx = {a: int(a.split("_")[1]) for a in agents}

    # Per-agent trajectories
    pending_log_probs: Dict[str, Optional[torch.Tensor]] = {
        a: None for a in agents
    }
    episode_log_probs: Dict[str, List[torch.Tensor]] = {a: [] for a in agents}
    episode_rewards: Dict[str, List[float]] = {a: [] for a in agents}

    total_return = 0.0

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        # Reward is for the *previous* action of this agent (if any).
        # Attach it to the last stored pending log_prob.
        if pending_log_probs[agent] is not None:
            episode_log_probs[agent].append(pending_log_probs[agent])
            episode_rewards[agent].append(float(reward))
            total_return += float(reward)
            pending_log_probs[agent] = None

        if termination or truncation:
            # Must still step with action=None once agent is done.
            env.step(None)
            continue

        # Observation -> tensor
        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)  # (1, 80)
        agent_idx = torch.tensor(
            [agent_to_idx[agent]], dtype=torch.long, device=device
        )  # (1,)

        # Action mask (binary 0/1 for each action)
        action_mask = np.asarray(info.get("action_mask", np.ones(5, dtype=np.int8)))
        if action_mask.sum() == 0:
            # Fallback: uniform legal actions if mask is somehow all zeros
            legal_actions = np.arange(env.action_space(agent).n)
        else:
            legal_actions = np.nonzero(action_mask)[0]

        logits = model(obs_t, agent_idx).squeeze(0)  # (5,)

        # Mask illegal actions by adding large negative value
        mask_t = torch.full_like(logits, -1e9)
        mask_t[torch.tensor(legal_actions, dtype=torch.long, device=device)] = 0.0
        masked_logits = logits + mask_t

        dist = torch.distributions.Categorical(logits=masked_logits)
        action_t = dist.sample()  # scalar tensor
        log_prob = dist.log_prob(action_t)

        # Store pending log_prob for when reward arrives next time this agent is active.
        pending_log_probs[agent] = log_prob

        env.step(int(action_t.item()))

    # Episode finished, collect any remaining pending log_probs (typically with zero reward)
    for agent in agents:
        if pending_log_probs[agent] is not None:
            episode_log_probs[agent].append(pending_log_probs[agent])
            episode_rewards[agent].append(0.0)
            pending_log_probs[agent] = None

    # Compute REINFORCE loss with discounted returns per agent and aggregate
    all_log_probs: List[torch.Tensor] = []
    all_returns: List[torch.Tensor] = []

    for agent in agents:
        rewards = episode_rewards[agent]
        if not rewards:
            continue

        # Discounted returns G_t
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = float(r) + gamma * G
            returns.append(G)
        returns.reverse()

        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns for variance reduction
        if returns_t.numel() > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        lp_stack = torch.stack(episode_log_probs[agent])  # (T,)

        all_log_probs.append(lp_stack)
        all_returns.append(returns_t)

    if not all_log_probs:
        return total_return

    log_probs_cat = torch.cat(all_log_probs)
    returns_cat = torch.cat(all_returns)

    loss = -(log_probs_cat * returns_cat).mean()

    model.zero_grad(set_to_none=True)
    loss.backward()

    # Return the loss so the caller can step the optimizer after this episode.
    return total_return, float(loss.detach().cpu().item())


def play_human_rendered_game(
    model: PolicyNetwork,
    device: torch.device,
    sleep_time: float = 0.5,
) -> None:
    """
    Play a single game with human rendering at normal speed for visualization.
    Uses greedy actions (argmax) under the learned policy.
    """
    env = make_ludo_env(mode="ffa", render_mode="human")
    env.reset()

    agents = env.possible_agents
    agent_to_idx = {a: int(a.split("_")[1]) for a in agents}

    print("Starting a human-rendered Ludo game with the trained policy...")

    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
            continue

        obs_np = np.asarray(obs, dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        agent_idx = torch.tensor(
            [agent_to_idx[agent]], dtype=torch.long, device=device
        )

        action_mask = np.asarray(info.get("action_mask", np.ones(5, dtype=np.int8)))
        if action_mask.sum() == 0:
            legal_actions = np.arange(env.action_space(agent).n)
        else:
            legal_actions = np.nonzero(action_mask)[0]

        with torch.no_grad():
            logits = model(obs_t, agent_idx).squeeze(0)

        mask_t = torch.full_like(logits, -1e9)
        mask_t[torch.tensor(legal_actions, dtype=torch.long, device=device)] = 0.0
        masked_logits = logits + mask_t

        action = int(torch.argmax(masked_logits).item())

        env.step(action)
        time.sleep(sleep_time)

    env.close()
    print("Human-rendered game finished.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a simple multi-agent self-play policy for Ludo."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100_000,
        help="Total number of training episodes (default: 100000).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "checkpoints"
        ),
        help="Directory to store checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1_000,
        help="How often (in episodes) to save checkpoints.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for returns (default: 0.99).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    # Rendering is disabled (render_mode=None) during training for speed.
    # If you want more speed, you can further optimize by using multiple
    # processes and aggregating gradients, but this script keeps things simple.
    env = make_ludo_env(mode="ffa", render_mode=None)

    # Seeding numpy and torch (PettingZoo env is seeded through env.reset(seed=...)).
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyNetwork(obs_dim=80, num_agents=4, n_actions=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_path = os.path.join(args.checkpoint_dir, "ludo_selfplay.pt")

    # Load checkpoint if present
    episodes_done = load_checkpoint(checkpoint_path, model, optimizer)
    if episodes_done > 0:
        print(f"Resuming training from checkpoint at episode {episodes_done}.")
    else:
        print("Starting training from scratch.")

    total_episodes = args.episodes

    try:
        for ep in range(episodes_done, total_episodes):
            # Optionally seed env per episode for reproducibility
            env.reset(seed=args.seed + ep)

            episode_return, episode_loss = run_episode(
                env, model, device, gamma=args.gamma
            )

            # Optimizer step after each episode
            optimizer.step()

            # Simple logging
            if (ep + 1) % 100 == 0 or ep == 0:
                print(
                    f"Episode {ep + 1}/{total_episodes} "
                    f"- Return: {episode_return:.2f} "
                    f"- Loss: {episode_loss:.4f}"
                )

            # Periodic checkpointing
            if (ep + 1) % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_path, model, optimizer, ep + 1)
                print(f"Checkpoint saved at episode {ep + 1}.")

        # Final checkpoint after training completes
        save_checkpoint(checkpoint_path, model, optimizer, total_episodes)
        print(f"Training finished. Final checkpoint saved at episode {total_episodes}.")

    except KeyboardInterrupt:
        # Save progress on interruption so we can resume later
        print("Training interrupted. Saving checkpoint...")
        current_ep = min(total_episodes, episodes_done)
        save_checkpoint(checkpoint_path, model, optimizer, current_ep)
        print(f"Checkpoint saved at episode {current_ep}.")
    finally:
        env.close()

    # After training, play one human-rendered game at normal speed
    play_human_rendered_game(model, device, sleep_time=0.5)


if __name__ == "__main__":
    main()

