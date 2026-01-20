# noqa: D212, D415
"""
# Ludo

Classic Ludo implemented as a PettingZoo AEC environment.
"""

from __future__ import annotations

import os
import numpy as np
import pygame
import gymnasium
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import agent_selector
from coordinates import *


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def get_image(path):
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, "img", path))
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "ludo_v0",
        "is_parallelizable": False,
        "render_fps": 4,
    }

    MAIN_TRACK_LEN = 52
    HOME_LEN = 6
    PIECES = 4

    SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}
    START_INDEX = [0, 13, 26, 39]

    def __init__(self, num_players=4, render_mode=None, screen_scaling=8):
        EzPickle.__init__(self, num_players, render_mode, screen_scaling)
        super().__init__()

        assert 2 <= num_players <= 4
        self.num_players = num_players
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}
        self.observation_spaces = {
            a: spaces.Dict(
                {
                    "observation": spaces.Box(0, 1, shape=(75,), dtype=np.float32),
                    "action_mask": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                }
            )
            for a in self.agents
        }

        self.screen = None
        self.clock = None

        self.reset()

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        self.np_random = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # piece_state[player][piece] = (zone, index)
        self.piece_state = {
            a: [("yard", None) for _ in range(self.PIECES)]
            for a in self.agents
        }

        # distance traveled on main track
        self.distance = {
            a: [0] * self.PIECES for a in self.agents
        }

        self.current_dice = 0

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------

    def observe(self, agent):
        obs = np.zeros(75, dtype=np.float32)

        # Main board occupancy
        board = np.zeros(self.MAIN_TRACK_LEN, dtype=np.float32)
        for a in self.agents:
            for zone, idx in self.piece_state[a]:
                if zone == "main":
                    board[idx] = 1.0
        obs[:52] = board

        offset = 52
        for a in self.agents:
            for zone, idx in self.piece_state[a]:
                if zone == "yard":
                    obs[offset] = 0.0
                elif zone == "main":
                    obs[offset] = (idx + 1) / 52.0
                elif zone == "home":
                    obs[offset] = 0.8 + idx / 10.0
                else:
                    obs[offset] = 1.0
                offset += 1

        obs[68] = self.current_dice / 6.0
        obs[69] = 1.0 if agent == self.agent_selection else 0.0

        action_mask = np.zeros(5, dtype=np.int8)
        if agent == self.agent_selection:
            for a in self._legal_actions(agent):
                action_mask[a] = 1

        return {"observation": obs, "action_mask": action_mask}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------

    def _legal_actions(self, agent):
        # If dice not rolled yet, only PASS is legal
        if self.current_dice == 0:
            return [4]

        legal = []
        for i, (zone, idx) in enumerate(self.piece_state[agent]):
            if zone == "finished":
                continue

            if zone == "yard" and self.current_dice == 6:
                legal.append(i)

            elif zone == "main":
                if self.distance[agent][i] + self.current_dice <= 51 + self.HOME_LEN:
                    legal.append(i)

            elif zone == "home":
                if idx + self.current_dice < self.HOME_LEN:
                    legal.append(i)

        if not legal:
            legal.append(4)

        return legal

    # ------------------------------------------------------------
    # Step
    # ------------------------------------------------------------

    def step(self, action):
        if (
            self.truncations[self.agent_selection]
            or self.terminations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        agent = self.agent_selection
        self.rewards = {a: 0 for a in self.agents}

        if self.current_dice == 0:
            self.current_dice = self.np_random.integers(1, 7)

        legal = self._legal_actions(agent)
        if action not in legal:
            return

        capture = False

        if action != 4:
            zone, idx = self.piece_state[agent][action]

            if zone == "yard":
                start = self.START_INDEX[self.agents.index(agent)]
                self.piece_state[agent][action] = ("main", start)
                self.distance[agent][action] = 0

            elif zone == "main":
                new_dist = self.distance[agent][action] + self.current_dice
                self.distance[agent][action] = new_dist

                if new_dist < 52:
                    new_pos = (idx + self.current_dice) % 52
                    self.piece_state[agent][action] = ("main", new_pos)
                    capture = self._check_capture(agent, new_pos)
                else:
                    self.piece_state[agent][action] = ("home", new_dist - 52)

            elif zone == "home":
                new_idx = idx + self.current_dice
                if new_idx == self.HOME_LEN - 1:
                    self.piece_state[agent][action] = ("finished", None)
                else:
                    self.piece_state[agent][action] = ("home", new_idx)

        # Win check
        if all(z == "finished" for z, _ in self.piece_state[agent]):
            for a in self.agents:
                self.terminations[a] = True
                self.rewards[a] = 1 if a == agent else -1
            self._accumulate_rewards()
            return

        extra_turn = self.current_dice == 6 or capture
        
        if self.render_mode == "human":
            self.render()
        
        self.current_dice = 0

        if not extra_turn:
            self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()

    # ------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------

    def _check_capture(self, agent, pos):
        if pos in self.SAFE_SQUARES:
            return False

        for a in self.agents:
            if a == agent:
                continue
            for i, (zone, idx) in enumerate(self.piece_state[a]):
                if zone == "main" and idx == pos:
                    self.piece_state[a][i] = ("yard", None)
                    self.distance[a][i] = 0
                    return True
        return False

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("Render called without render_mode.")
            return

        size = 800
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("Ludo")
                self.screen = pygame.display.set_mode((size, size))
            else:
                self.screen = pygame.Surface((size, size))
            self.clock = pygame.time.Clock()

        self.screen.fill((255, 255, 255))
        board_img = get_image("Board_Main.png")
        original_board_size = board_img.get_width()  # 1804x1804
        scale = size / original_board_size
        board_img = pygame.transform.scale(board_img, (size, size))
        self.screen.blit(board_img, (0, 0))

        if self.current_dice > 0:
            dice_img = get_image(f"Dice_{self.current_dice}.png")
            dice_original_size = dice_img.get_size()  # (180, 189)
            dice_scaled_size = (int(dice_original_size[0] * scale), int(dice_original_size[1] * scale))
            dice_img = pygame.transform.scale(dice_img, dice_scaled_size)
            self.screen.blit(dice_img, (10, 10))

        # Render pieces
        piece_colors = {"player_0": "Green", "player_1": "Yellow", "player_2": "Blue", "player_3": "Red"}
        for agent in self.agents:
            piece_img = get_image(f"Piece_{piece_colors[agent]}.png")
            piece_original_size = piece_img.get_size()
            piece_scaled_size = (int(piece_original_size[0] * scale), int(piece_original_size[1] * scale))
            piece_img = pygame.transform.scale(piece_img, piece_scaled_size)
            piece_offset_x = piece_scaled_size[0] // 2
            piece_offset_y = piece_scaled_size[1] // 2

            for piece_idx, (zone, idx) in enumerate(self.piece_state[agent]):
                if zone == "yard":
                    x, y = YARD_POSITIONS[agent][piece_idx]
                elif zone == "main":
                    x, y = MAIN_TRACK_POSITIONS[idx]
                elif zone in ("home", "finished"):
                    # For finished pieces, always use the final home index
                    home_idx = idx if zone == "home" else self.HOME_LEN - 1
                    x, y = HOME_TRACK_POSITIONS[agent][home_idx]
                else:
                    continue

                screen_x = int(x * scale)
                screen_y = int(y * scale)
                self.screen.blit(piece_img, (screen_x - piece_offset_x, screen_y - piece_offset_y))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        obs = np.array(pygame.surfarray.pixels3d(self.screen))
        return np.transpose(obs, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
