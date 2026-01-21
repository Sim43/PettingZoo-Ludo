import torch

from single import (
    ActorCritic,
    CHECKPOINT_PATH,
    DEVICE,
    ludo_env,
    run_episode_collect,
)


def eval_game():
    model = ActorCritic().to(DEVICE)
    payload = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload["model"])

    eval_env = ludo_env(render_mode="human")
    _ = run_episode_collect(eval_env, model, render=True)
    eval_env.close()


if __name__ == "__main__":
    eval_game()