from ludo import env
import random

game = env(render_mode="human")
game.reset()

print("Starting Ludo test")

while game.agents:
    agent = game.agent_selection
    obs = game.observe(agent)

    mask = obs["action_mask"]
    legal = [i for i, v in enumerate(mask) if v == 1]

    print(f"\n{agent}")
    print("Dice:", int(obs["observation"][68] * 6))
    print("Legal actions:", legal)

    if not legal:
        action = 4  # forced pass
    else:
        action = random.choice(legal)

    print("Action:", action)
    game.step(action)

    if game.terminations[agent]:
        print("Winner:", agent)
        break

game.close()
