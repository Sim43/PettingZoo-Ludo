from ludo.ludo import env
import random
import time

game = env(render_mode="human")
game.reset()

print("Starting Ludo test")

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
        print(f"\n🏆 Winner: {agent}")
        break

    time.sleep(0.2)

game.close()
print("\nTest finished.")
