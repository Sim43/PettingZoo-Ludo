import os
import torch

from single import (
    ActorCritic,
    BEST_CHECKPOINT_PATH,
    CHECKPOINT_PATH,
    DEVICE,
    ludo_env,
    run_episode_collect,
)


def eval_game():
    model = ActorCritic().to(DEVICE)
    
    # Use best checkpoint if available, otherwise fall back to latest
    checkpoint_path = BEST_CHECKPOINT_PATH if os.path.exists(BEST_CHECKPOINT_PATH) else CHECKPOINT_PATH
    if checkpoint_path == CHECKPOINT_PATH:
        print("Best checkpoint not found, using latest checkpoint")
    
    payload = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(payload["model"])
    episode = payload.get("episode_count", 0)
    print(f"Loaded checkpoint from episode {episode}")

    eval_env = ludo_env(render_mode="human")
    _ = run_episode_collect(eval_env, model, render=True)
    eval_env.close()


if __name__ == "__main__":
    eval_game()