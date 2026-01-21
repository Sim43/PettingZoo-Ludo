# noqa: D212, D415
"""
# Ludo

```{figure} ludo_board.png
:width: 140px
:name: ludo
```

This environment is a turn-based, multi-player board game environment compatible with the PettingZoo AEC API.

| Import             | `from ludo import env`                             |
|--------------------|----------------------------------------------------|
| Actions            | Discrete                                           |
| Parallel API       | No                                                 |
| Manual Control     | No                                                 |
| Agents             | `agents = ['player_0', 'player_1', 'player_2', 'player_3']` |
| Agents             | 2–4 (configurable)                                 |
| Action Shape       | (1,)                                               |
| Action Values      | Discrete(5)                                        |
| Observation Shape  | (80,)                                              |
| Observation Values | [0, 1]                                             |

Ludo is a classic 2–4 player race game. Each player has four pieces which start in a yard and must travel once around the shared main track and then along a player-specific home track. Players take turns rolling a single six-sided die and moving one of their pieces according to the roll. A roll of
6 brings a piece out of the yard and also grants an extra turn.

### Game Modes

The environment supports two modes via the `mode` parameter:

* **Free-for-all** (`mode="ffa"`, default): Each player competes individually. The first player to bring all four pieces to their final home position wins.

* **Teams** (`mode="teams"`): Fixed 2v2 team-based play. Teams: (player_0, player_2) vs (player_1, player_3). A team wins when both teammates finish all four pieces. Teammates cannot capture each other but can form team blocks (2+ pieces from the same team on a square). Finished agents can use their dice rolls to move their teammate's pieces (dice-sharing).

### Observation Space

The observation is a **flat numpy array of length 80** (`dtype=np.float32`):

* **Indices 0–74**: Core game state encoding:
  * **0–51**: Main-track occupancy (52 shared squares).
  * **52–67**: Each piece's zone (yard / main / home / finished) and progress across all players.
  * **68**: Normalized dice value (`dice / 6.0`).
  * **69**: `1.0` if it is this agent's turn, `0.0` otherwise.
* **Indices 75–79**: Action mask (binary, `1.0` = legal, `0.0` = illegal) for actions 0–4.

The action mask is also exposed in `info["action_mask"]` as `np.int8` for compatibility with PettingZoo wrappers. Only the currently acting agent has a non-zero action mask. All other agents receive an all-zero mask.

#### Legal Actions Mask

Legal moves for the current agent are given by the action mask (indices 75–79). The action space is `Discrete(5)`:

* `0–3`: move the corresponding piece index (if legal),
* `4`: PASS (only legal when no movement actions are available).

In teams mode, if an agent has all pieces finished, their action mask reflects their teammate's legal moves (dice-sharing).

Any action index with mask value 0 is illegal and, when taken, will terminate the episode for the acting agent via `TerminateIllegalWrapper`.

### Action Space

The action space is the set of integers from 0 to 4 (inclusive). On each turn, the dice has already been rolled, and the agent chooses which piece to move (or is forced to PASS when no moves are available).

### Rewards

**Free-for-all mode:**
* Winning agent: +1
* Losing agents: -1
* Illegal move: -1 for the acting agent (via wrapper), 0 for others
* All other intermediate moves: 0

**Teams mode:**
* Winning team (both teammates): +1 for each agent on the winning team
* Losing team: -1 for each agent on the losing team
* Illegal move: -1 for the acting agent (via wrapper), 0 for others
* All other intermediate moves: 0

### Version History

* v0: Initial release.
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
from ludo.coordinates import *


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


# Team mapping for 2v2 mode
TEAM_MAP = {
    "player_0": 0,
    "player_2": 0,
    "player_1": 1,
    "player_3": 1,
}


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

class raw_env(AECEnv, EzPickle):
    """
    Ludo environment supporting both free-for-all (FFA) and 2v2 team-based modes.
    
    Modes:
    - mode="ffa" (default): Classic free-for-all where each player competes individually.
      First player to finish all pieces wins.
    - mode="teams": 2v2 cooperative-competitive mode. Teams: (player_0, player_2) vs (player_1, player_3).
      Team wins when both teammates finish all pieces. Teammates cannot capture each other but can form blocks.
      Finished agents can use their dice rolls to move their teammate's pieces (dice-sharing).
    """
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

    def __init__(self, num_players=4, render_mode=None, screen_scaling=8, mode="ffa"):
        EzPickle.__init__(self, num_players, render_mode, screen_scaling, mode)
        super().__init__()

        assert 2 <= num_players <= 4
        assert mode in ("ffa", "teams"), f"mode must be 'ffa' or 'teams', got '{mode}'"
        if mode == "teams":
            assert num_players == 4, "teams mode requires exactly 4 players"
        
        self.num_players = num_players
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling
        self.mode = mode
        self.team_mode = (mode == "teams")
        
        # Team mapping: only used in teams mode
        if self.team_mode:
            self.team_map = {a: TEAM_MAP[a] for a in [f"player_{i}" for i in range(4)]}
        else:
            self.team_map = {}

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]

        # 5 actions: move piece 0-3, 4 = PASS (only legal when no moves exist)
        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}
        # Flattened observation: 75-element state vector + 5-element action mask
        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, shape=(80,), dtype=np.float32)
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
        # Track consecutive sixes per player for three-sixes penalty
        self.consecutive_sixes = {a: 0 for a in self.agents}

        # Track whether each colour (agent) has captured at least one enemy piece.
        # This gates home entry per colour in both FFA and teams modes.
        self.has_captured = {a: False for a in self.agents}

        # In free-for-all mode, track the order in which players finish all pieces.
        # Used to assign rank-based rewards at the end of the episode.
        self.finish_order = []

        # Roll dice for the first agent so there is always an active dice value
        self._roll_new_dice()

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------

    def observe(self, agent):
        # Core state features
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
                    # Encode home progress in (0.8, 1.0) and stay within [0, 1]
                    obs[offset] = 0.8 + idx / 30.0
                else:
                    obs[offset] = 1.0
                offset += 1

        obs[68] = self.current_dice / 6.0
        obs[69] = 1.0 if agent == self.agent_selection else 0.0

        # Action mask as a separate vector, then concatenated to the observation.
        # Gymnasium's Discrete space expects mask dtype to be np.int8.
        action_mask = np.zeros(5, dtype=np.int8)
        if agent == self.agent_selection:
            # In teams mode: if agent is finished, check teammate's legal actions
            if self.team_mode:
                all_finished = all(z == "finished" for z, _ in self.piece_state[agent])
                if all_finished:
                    teammate = self._get_teammate(agent)
                    if teammate is not None:
                        # Agent helps teammate: use teammate's legal actions
                        legal = self._legal_actions(agent, target_agent=teammate)
                    else:
                        legal = [4]  # No teammate, only PASS
                else:
                    legal = self._legal_actions(agent)
            else:
                legal = self._legal_actions(agent)
            
            for a in legal:
                action_mask[a] = 1

        # Expose the action mask through info so TerminateIllegalWrapper can use it
        self.infos[agent]["action_mask"] = action_mask

        # Return a single flat numpy array to satisfy PettingZoo's preferred API
        return np.concatenate([obs, action_mask.astype(np.float32)])

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------
    # Team helpers
    # ------------------------------------------------------------

    def _same_team(self, a, b):
        """Return True if agents a and b are on the same team (teams mode only)."""
        if not self.team_mode:
            return False
        return self.team_map.get(a) == self.team_map.get(b)

    def _get_teammate(self, agent):
        """Return the teammate of agent (teams mode only). Returns None if not in teams mode or no teammate."""
        if not self.team_mode:
            return None
        team_id = self.team_map.get(agent)
        if team_id is None:
            return None
        for a in self.agents:
            if a != agent and self.team_map.get(a) == team_id:
                return a
        return None

    def _is_enemy(self, agent, other):
        """Return True if other is an enemy of agent (different team in teams mode, different agent in FFA)."""
        if not self.team_mode:
            return agent != other
        return not self._same_team(agent, other)

    # ------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------

    def _pieces_on_main(self, pos):
        """Return list of (agent, piece_idx) on a given main-track position."""
        pieces = []
        for a in self.agents:
            for i, (zone, idx) in enumerate(self.piece_state[a]):
                if zone == "main" and idx == pos:
                    pieces.append((a, i))
        return pieces

    def _is_any_block(self, pos):
        """Return True if any block (2+ aligned pieces) is on this main-track position.

        - FFA: blocks are per colour (2+ pieces of the same agent).
        - Teams mode: blocks are per team (2+ pieces from the same team, possibly split across teammates).
        """
        pieces = self._pieces_on_main(pos)
        if not self.team_mode:
            # Free-for-all: count per agent/colour.
            counts = {}
            for a, _ in pieces:
                counts[a] = counts.get(a, 0) + 1
            return any(c >= 2 for c in counts.values())
        else:
            # Teams: count per team id so teammates can form team blocks.
            team_counts = {}
            for a, _ in pieces:
                team_id = self.team_map.get(a)
                if team_id is None:
                    continue
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            return any(c >= 2 for c in team_counts.values())

    def _is_enemy_block(self, agent, pos):
        """Return True if an enemy block (2+ aligned pieces) is on this main-track position."""
        pieces = self._pieces_on_main(pos)
        if not self.team_mode:
            # FFA: count per enemy colour.
            counts = {}
            for a, _ in pieces:
                if not self._is_enemy(agent, a):
                    continue
                counts[a] = counts.get(a, 0) + 1
            return any(c >= 2 for c in counts.values())
        else:
            # Teams: count per enemy team id.
            agent_team = self.team_map.get(agent)
            team_counts = {}
            for a, _ in pieces:
                team_id = self.team_map.get(a)
                if team_id is None or team_id == agent_team:
                    continue
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            return any(c >= 2 for c in team_counts.values())

    def _is_enemy_occupied(self, agent, pos):
        """Check if any enemy piece is on a given main-track position."""
        for a in self.agents:
            if not self._is_enemy(agent, a):
                continue
            for zone, idx in self.piece_state[a]:
                if zone == "main" and idx == pos:
                    return True
        return False

    def _legal_actions(self, agent, target_agent=None):
        """
        Return legal actions for agent.
        In teams mode, if target_agent is provided, compute legal actions for target_agent's pieces
        (used when finished agent helps teammate).
        """
        # Determine which agent's pieces we're checking
        check_agent = target_agent if target_agent is not None else agent
        
        legal = []
        for i, (zone, idx) in enumerate(self.piece_state[check_agent]):
            if zone == "finished":
                continue

            if zone == "yard" and self.current_dice == 6:
                # Entering main track:
                # - On non-safe squares: blocked by any block.
                # - On safe squares: allow stacking of any colours, do not block entry.
                start = self.START_INDEX[self.agents.index(check_agent)]
                if start in self.SAFE_SQUARES:
                    blocked_entry = False
                else:
                    blocked_entry = self._is_any_block(start)
                if not blocked_entry:
                    legal.append(i)

            elif zone == "main":
                # Main-track movement behaves differently before and after this colour
                # has captured at least one enemy piece.
                if not self.has_captured[check_agent]:
                    # Pre-capture: home entry is blocked; movement always stays on main track and
                    # may wrap around using modulo arithmetic. We only prevent passage through
                    # blocked non-safe squares.
                    blocked = False
                    for step in range(1, self.current_dice + 1):
                        pos = (idx + step) % self.MAIN_TRACK_LEN
                        # On safe squares, allow stacking of any colours (no blocking, no captures).
                        if pos in self.SAFE_SQUARES:
                            continue
                        if self._is_any_block(pos):
                            blocked = True
                            break
                    if not blocked:
                        legal.append(i)
                else:
                    # Post-capture: existing behaviour. Moves that reach or cross home entry
                    # must enter the home track; wrapping is disallowed.
                    new_dist = self.distance[check_agent][i] + self.current_dice
                    # Maximum total distance: last main square (distance MAIN_TRACK_LEN - 2)
                    # plus reachable home track (indices 0 .. HOME_LEN-1).
                    if new_dist <= (self.MAIN_TRACK_LEN - 2) + (self.HOME_LEN - 1):
                        # Check path for any blocks / occupied safe squares (cannot land on or pass through)
                        blocked = False
                        for step in range(1, self.current_dice + 1):
                            dist = self.distance[check_agent][i] + step
                            # Once we cross the last main square (distance MAIN_TRACK_LEN - 2),
                            # remaining movement is inside home track.
                            if dist > self.MAIN_TRACK_LEN - 2:
                                break  # into home track, no more main squares
                            pos = (idx + step) % self.MAIN_TRACK_LEN
                            # On safe squares, allow stacking of any colours (no blocking, no captures).
                            if pos in self.SAFE_SQUARES:
                                continue
                            if self._is_any_block(pos):
                                blocked = True
                                break
                        if not blocked:
                            legal.append(i)

            elif zone == "home":
                if idx + self.current_dice < self.HOME_LEN:
                    legal.append(i)

        # If no moves are possible, only PASS (4) is legal
        if not legal:
            return [4]

        return legal

    # ------------------------------------------------------------
    # Dice handling
    # ------------------------------------------------------------

    def _roll_new_dice(self):
        """Roll a new dice for the current agent and handle three-sixes penalty."""
        while True:
            agent = self.agent_selection
            roll = self.np_random.integers(1, 7)
            self.current_dice = roll
            if roll == 6:
                self.consecutive_sixes[agent] += 1
            else:
                self.consecutive_sixes[agent] = 0

            # Three consecutive sixes penalty: skip this agent's turn, no move.
            if self.consecutive_sixes[agent] >= 3:
                self.consecutive_sixes[agent] = 0
                self.current_dice = 0
                # Advance to next agent and immediately roll for them, without recursion.
                self.agent_selection = self._agent_selector.next()
                continue

            # Normal case: stop after a valid roll that does not trigger the penalty.
            break

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

        # In teams mode: check if agent is finished and helping teammate
        target_agent = agent
        action_piece_idx = action
        all_finished = False
        if self.team_mode:
            all_finished = all(z == "finished" for z, _ in self.piece_state[agent])
            if all_finished:
                teammate = self._get_teammate(agent)
                if teammate is not None:
                    # Agent is helping teammate: remap action to teammate's pieces
                    target_agent = teammate
                    # Legal actions are computed for teammate's pieces, so action index is already correct
                    action_piece_idx = action

        legal = self._legal_actions(agent, target_agent=target_agent if self.team_mode and all_finished else None)

        # With PASS action, there is always at least one legal action (4) even if no moves exist
        if action not in legal:
            # Illegal move -> let TerminateIllegalWrapper handle termination/penalty
            return

        capture = False
        finished_this_move = False

        # PASS action: skip movement, advance turn and roll new dice
        if action == 4:
            self.agent_selection = self._agent_selector.next()
            self._roll_new_dice()
            self._accumulate_rewards()
            return

        zone, idx = self.piece_state[target_agent][action_piece_idx]

        if zone == "yard":
            start = self.START_INDEX[self.agents.index(target_agent)]
            self.piece_state[target_agent][action_piece_idx] = ("main", start)
            self.distance[target_agent][action_piece_idx] = 0

        elif zone == "main":
            roll = self.current_dice
            if not self.has_captured[target_agent]:
                # Pre-capture: home entry is blocked; always stay on main track and wrap using modulo.
                new_pos = (idx + roll) % self.MAIN_TRACK_LEN
                self.piece_state[target_agent][action_piece_idx] = ("main", new_pos)
                capture = self._check_capture(target_agent, new_pos)
            else:
                # Post-capture: existing behaviour. Moves that reach or cross home entry
                # enter the home track; wrapping is disallowed.
                # Move along the main track, then into the home track (if needed) in a single move.
                # This prevents skipping the home entry or remaining on the main track after passing it.
                current_dist = self.distance[target_agent][action_piece_idx]
                new_dist = current_dist + roll
                self.distance[target_agent][action_piece_idx] = new_dist

                last_main_distance = self.MAIN_TRACK_LEN - 2  # color-specific last main index (50, 11, 24, 37)
                if new_dist <= last_main_distance:
                    # Entire move stays on the main track.
                    new_pos = (idx + roll) % self.MAIN_TRACK_LEN
                    self.piece_state[target_agent][action_piece_idx] = ("main", new_pos)
                    capture = self._check_capture(target_agent, new_pos)
                else:
                    # The move crosses the home entry: consume remaining steps on the main track,
                    # then move the remainder inside the home track within this single move.
                    # Steps needed to reach the last main-track index (distance last_main_distance).
                    steps_to_entry = last_main_distance - current_dist
                    steps_in_home = roll - steps_to_entry - 1  # first step after entry is home index 0
                    # If we reach or pass the final home index in this single move, mark as finished.
                    if steps_in_home >= self.HOME_LEN - 1:
                        self.piece_state[target_agent][action_piece_idx] = ("finished", None)
                        finished_this_move = True
                    else:
                        # Otherwise, land somewhere on the home track.
                        self.piece_state[target_agent][action_piece_idx] = ("home", steps_in_home)

        elif zone == "home":
            new_idx = idx + self.current_dice
            if new_idx == self.HOME_LEN - 1:
                self.piece_state[target_agent][action_piece_idx] = ("finished", None)
                finished_this_move = True
            else:
                self.piece_state[target_agent][action_piece_idx] = ("home", new_idx)

        # Win check
        if self.team_mode:
            # Teams mode: check if both teammates have finished all pieces
            # Check the team of the agent whose piece just moved (target_agent)
            team_id = self.team_map.get(target_agent)
            if team_id is not None:
                team_agents = [a for a in self.agents if self.team_map.get(a) == team_id]
                team_finished = all(
                    all(z == "finished" for z, _ in self.piece_state[a])
                    for a in team_agents
                )
                if team_finished:
                    # Winning team gets +1, losing team gets -1
                    for a in self.agents:
                        self.terminations[a] = True
                        if self.team_map.get(a) == team_id:
                            self.rewards[a] = 1
                        else:
                            self.rewards[a] = -1
                    self._accumulate_rewards()
                    return
        else:
            # FFA mode: rank-based terminal rewards (1st, 2nd, 3rd, last).
            # We do NOT end the game as soon as the first player finishes.
            # Instead:
            # - Track the order in which players finish all pieces.
            # - Continue play until all but one player have finished.
            # - Then assign position-based rewards and terminate.
            if all(z == "finished" for z, _ in self.piece_state[agent]):
                # Record this agent's finishing position if not already recorded.
                if agent not in self.finish_order:
                    self.finish_order.append(agent)

                # Once all but one player have finished, determine the last player
                # and assign rewards by final rank.
                if len(self.finish_order) >= self.num_players - 1:
                    # The remaining player (who never finished) is last.
                    remaining = [a for a in self.agents if a not in self.finish_order]
                    if remaining:
                        self.finish_order.extend(remaining)

                    # Fixed rewards for up to 4 positions:
                    # 1st:  +1.0
                    # 2nd:  +0.3
                    # 3rd:  -0.3
                    # 4th+: -1.0
                    rank_rewards = [1.0, 0.3, -0.3, -1.0]

                    for idx, a in enumerate(self.finish_order):
                        self.terminations[a] = True
                        reward = (
                            rank_rewards[idx]
                            if idx < len(rank_rewards)
                            else rank_rewards[-1]
                        )
                        self.rewards[a] = reward

                    # Any agent not in finish_order (should not happen in normal play)
                    # is treated as last place.
                    for a in self.agents:
                        if a not in self.finish_order:
                            self.terminations[a] = True
                            self.rewards[a] = rank_rewards[-1]

                    self._accumulate_rewards()
                    return

        # Extra turn conditions:
        # - rolling a 6
        # - capturing an enemy piece
        # - finishing a piece (entering the final home position this move)
        extra_turn = self.current_dice == 6 or capture or finished_this_move

        if self.render_mode == "human":
            self.render()

        if extra_turn:
            # Same agent gets another turn with a new dice
            self._roll_new_dice()
        else:
            # Next agent's turn
            self.agent_selection = self._agent_selector.next()
            self._roll_new_dice()

        self._accumulate_rewards()

    # ------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------

    def _check_capture(self, agent, pos):
        # Entry squares and other safe squares: no captures allowed
        if pos in self.SAFE_SQUARES:
            return False

        # Any block (friendly or enemy) cannot be captured
        if self._is_any_block(pos):
            return False

        # Count opponent pieces on this square
        pieces = self._pieces_on_main(pos)
        counts = {}
        indices = {}
        for a, i in pieces:
            # In teams mode: skip teammates; in FFA: skip self
            if not self._is_enemy(agent, a):
                continue
            counts[a] = counts.get(a, 0) + 1
            indices.setdefault(a, []).append(i)

        # No opponent pieces
        if not counts:
            return False

        # Single opponent piece: capture it (blocks already excluded above)
        for a, c in counts.items():
            if c == 1:
                i = indices[a][0]
                self.piece_state[a][i] = ("yard", None)
                self.distance[a][i] = 0
                # Mark that this colour has captured at least one enemy piece.
                self.has_captured[agent] = True
                # After the first capture for this agent, recompute main-track distances
                # so that subsequent home-entry behaviour matches the standard rules.
                start_idx = self.START_INDEX[self.agents.index(agent)]
                for p_idx, (z, idx2) in enumerate(self.piece_state[agent]):
                    if z == "main" and idx2 is not None:
                        self.distance[agent][p_idx] = (idx2 - start_idx) % self.MAIN_TRACK_LEN
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
                # Small deterministic offset only when multiple pieces share the same logical position.
                # For finished pieces, we also apply offsets so stacked finished pieces are visible.
                same_spot_count = 0
                for a2 in self.agents:
                    for _, (z2, idx2) in enumerate(self.piece_state[a2]):
                        if z2 != zone:
                            continue
                        if zone == "finished":
                            # All finished pieces of the same colour share the final home position.
                            if a2 == agent:
                                same_spot_count += 1
                        else:
                            # For yard/main/home we require a concrete matching index.
                            if idx2 is not None and idx2 == idx:
                                same_spot_count += 1
                if same_spot_count > 1:
                    stack_offset_x = (piece_idx % 2) * 6
                    stack_offset_y = (piece_idx // 2) * 6
                else:
                    stack_offset_x = 0
                    stack_offset_y = 0
                self.screen.blit(
                    piece_img,
                    (
                        screen_x - piece_offset_x + stack_offset_x,
                        screen_y - piece_offset_y + stack_offset_y,
                    ),
                )

        # Render capture markers (one per colour) at the precomputed yard centers
        # when that colour has captured at least one enemy piece.
        for agent in self.agents:
            if getattr(self, "has_captured", {}).get(agent, False):
                cx, cy = CAPTURE_MARK_POSITIONS[agent]
                screen_cx = int(cx * scale)
                screen_cy = int(cy * scale)
                # Small coloured circle as an always-on marker (acts like an emoji).
                colour_map = {
                    "player_0": (0, 200, 0),      # Green
                    "player_1": (220, 200, 0),    # Yellow
                    "player_2": (0, 0, 220),      # Blue
                    "player_3": (200, 0, 0),      # Red
                }
                pygame.draw.circle(self.screen, colour_map[agent], (screen_cx, screen_cy), 10)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        obs = np.array(pygame.surfarray.pixels3d(self.screen))
        return np.transpose(obs, (1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
