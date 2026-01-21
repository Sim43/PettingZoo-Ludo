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

    # Default is single mode unless --team is explicitly requested
    mode = "single"
    if args.team:
        mode = "teams"

    game = env(render_mode="human", mode=mode)
    game.reset()

    print(f"Starting Ludo test in mode='{mode}'")

    MAX_STEPS = 1000
    steps = 0

    while game.agents and steps < MAX_STEPS:
        steps += 1

        agent = game.agent_selection
        obs = game.observe(agent)

        # Last 5 entries are the action mask, first 75 are the core observation
        mask = obs[75:]
        legal = [i for i, v in enumerate(mask) if v == 1]
        dice = int(obs[68] * 6)

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

        if game.terminations[agent]:
            # In teams mode, both teammates win together. Show all agents with positive reward.
            if mode == "teams":
                winners = [a for a, r in game.rewards.items() if r > 0]
                print(f"\n🏆 Winning team: {winners}")
            else:
                print(f"\n🏆 Winner: {agent}")
            break

        time.sleep(0.1)

    game.close()
    print("\nTest finished.")


if __name__ == "__main__":
    main()
