import os
import argparse
import numpy as np
import torch

from single import (
    ActorCritic,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    DEVICE,
    ludo_env,
    run_episode_collect,
    evaluate,
    load_checkpoint,
)


def eval_game(model: ActorCritic, num_games: int = 1, render: bool = True, verbose: bool = True):
    """Run evaluation games with the model."""
    eval_env = ludo_env(render_mode="human" if render else None)
    
    wins_per_agent = {}
    returns_per_agent = {}
    episode_lengths = []
    
    for game_num in range(num_games):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Game {game_num + 1}/{num_games}")
            print(f"{'='*60}")
        
        data = run_episode_collect(eval_env, model, render=render)
        
        # Compute statistics for this game
        episode_returns = {}
        for agent, traj in data.items():
            if agent not in returns_per_agent:
                returns_per_agent[agent] = []
                wins_per_agent[agent] = 0
            
            if traj:
                agent_return = sum(t.reward for t in traj)
                episode_returns[agent] = agent_return
                returns_per_agent[agent].append(agent_return)
        
        # Determine winner(s) - highest return
        if episode_returns:
            max_return = max(episode_returns.values())
            winners = [a for a, r in episode_returns.items() if r == max_return and r > 0]
            for winner in winners:
                wins_per_agent[winner] = wins_per_agent.get(winner, 0) + 1
            
            episode_length = sum(len(traj) for traj in data.values())
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"\nGame Results:")
                print(f"  Episode length: {episode_length} steps")
                print(f"  Returns:")
                for agent in sorted(episode_returns.keys()):
                    marker = " 🏆" if agent in winners else ""
                    print(f"    {agent}: {episode_returns[agent]:.3f}{marker}")
                print(f"  Winner(s): {', '.join(winners) if winners else 'None'}")
    
    eval_env.close()
    
    # Print summary statistics
    if verbose and num_games > 1:
        print(f"\n{'='*60}")
        print(f"Summary Statistics ({num_games} games):")
        print(f"{'='*60}")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f} steps")
        print(f"  Win rates:")
        for agent in sorted(wins_per_agent.keys()):
            win_rate = wins_per_agent[agent] / num_games
            avg_return = np.mean(returns_per_agent[agent]) if returns_per_agent[agent] else 0.0
            print(f"    {agent}: {win_rate:.1%} ({wins_per_agent[agent]}/{num_games} wins), "
                  f"avg return: {avg_return:.3f}")
        print(f"{'='*60}\n")
    
    return {
        'wins': wins_per_agent,
        'returns': returns_per_agent,
        'episode_lengths': episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Ludo model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=f"Path to checkpoint file (default: {BEST_CHECKPOINT_PATH} if exists, else {CHECKPOINT_PATH})"
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering (faster evaluation)"
    )
    parser.add_argument(
        "--eval-stats",
        action="store_true",
        help="Run statistical evaluation (20 episodes, no rendering)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed information (default: True)"
    )
    args = parser.parse_args()
    
    model = ActorCritic().to(DEVICE)
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif os.path.exists(BEST_CHECKPOINT_PATH):
        checkpoint_path = BEST_CHECKPOINT_PATH
        if args.verbose:
            print(f"Using best checkpoint: {BEST_CHECKPOINT_PATH}")
    elif os.path.exists(CHECKPOINT_PATH):
        checkpoint_path = CHECKPOINT_PATH
        if args.verbose:
            print(f"Using latest checkpoint: {CHECKPOINT_PATH}")
    else:
        print(f"Error: No checkpoint found at {BEST_CHECKPOINT_PATH} or {CHECKPOINT_PATH}")
        print("Please train a model first or specify --checkpoint path")
        return
    
    # Load checkpoint
    episode = load_checkpoint(model, None, None, checkpoint_path)
    if args.verbose:
        print(f"Loaded checkpoint from episode {episode}")
        print(f"Checkpoint path: {checkpoint_path}\n")
    
    # Run evaluation
    if args.eval_stats:
        # Statistical evaluation (no rendering, multiple episodes)
        if args.verbose:
            print("Running statistical evaluation...")
        stats = evaluate(model, num_episodes=20)
        print(f"\n{'='*60}")
        print("Evaluation Statistics (20 episodes):")
        print(f"{'='*60}")
        print(f"  Average episode length: {stats['avg_episode_length']:.1f} steps")
        print(f"  Overall win rate: {stats['win_rate']:.3f}")
        print(f"  Average return: {stats['avg_return']:.3f}")
        print(f"\n  Per-agent statistics:")
        for agent in ['player_0', 'player_1', 'player_2', 'player_3']:
            if f'{agent}_avg_return' in stats:
                print(f"    {agent}:")
                print(f"      Average return: {stats[f'{agent}_avg_return']:.3f}")
                print(f"      Win rate: {stats[f'{agent}_win_rate']:.3f}")
        print(f"{'='*60}\n")
    else:
        # Interactive evaluation (with rendering)
        eval_game(
            model,
            num_games=args.num_games,
            render=not args.no_render,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
