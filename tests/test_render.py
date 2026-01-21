from ludo.ludo import env
import random
import time
import argparse


def main():
    parser = argparse.ArgumentParser(description="Render test for Ludo environment.")
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run in single-player single mode (default).",
    )
    parser.add_argument(
        "--team",
        action="store_true",
        help="Run in 2v2 team mode.",
    )
    args = parser.parse_args()

    # Default is single/single mode unless --team is explicitly requested
    mode = "single"
    if args.team:
        mode = "teams"

    game = env(render_mode="human", mode=mode)
    game.reset(seed=43)

    print(f"Starting Ludo test in mode='{mode}'")

    MAX_STEPS = 1000
    steps = 0

    # Track full-episode returns by summing the per-step rewards dict, which is
    # exactly what PettingZoo's api_test uses to reconstruct rewards.
    episode_returns = {a: 0.0 for a in game.agents}

    while game.agents and steps < MAX_STEPS:
        steps += 1

        agent = game.agent_selection
        obs = game.observe(agent)

        # Last 5 entries are the action mask, first 75 are the core observation.
        # Keep this aligned with the environment's documented observation layout.
        mask = obs[75:]
        legal = [i for i, v in enumerate(mask) if v == 1]
        # Use the environment's internal dice state directly to avoid any
        # decoding/rounding issues from the observation.
        dice = getattr(game, "current_dice", 0)

        print(f"\nAgent: {agent}")
        print(f"Dice: {dice}")
        print(f"Legal actions: {legal}")

        if legal:
            action = random.choice(legal)
            print(f"Chosen action: {action}")
        else:
            # Must pass a VALID action, env will auto-pass
            action = 0
            print("No legal moves -> implicit pass")

        game.step(action)

        # Accumulate per-step rewards into episode returns so that dense shaping
        # and penalties across the whole game are reflected in the final totals.
        for a, r in game.rewards.items():
            episode_returns[a] = episode_returns.get(a, 0.0) + r

        if game.terminations[agent]:
            # At the end of the game, show per-player episode returns so that
            # both dense shaping and terminal rewards are visible.
            rewards_source = episode_returns
            standings = sorted(
                rewards_source.items(), key=lambda x: x[1], reverse=True
            )

            if mode == "teams":
                print("\n🏆 Final standings (teams, per-player rewards):")
            else:
                print("\n🏁 Final standings (single, per-player rewards):")

            for rank, (a, r) in enumerate(standings, start=1):
                print(f"{rank}. {a} (reward={r})")
            break

        time.sleep(0.1)

    game.close()
    print("\nTest finished.")


if __name__ == "__main__":
    main()
