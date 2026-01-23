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
| Action Values      | Discrete(65) = (4 pieces × 16 dice slots) + 1 PASS |
| Observation Shape  | (86,)                                              |
| Observation Values | [0, 1]                                             |

Ludo is a classic 2–4 player race game. Each player has four pieces which start in a yard and must travel once around the shared main track and then along a player-specific home track. Players use **dice banking**: rolling a 6 adds it to their bank and grants another roll; non-6 dice are also banked. Players explicitly choose which die from their bank to use for each move. Rolling three consecutive 6s cancels that roll attempt but preserves previously banked dice. Finishing a piece, capturing an enemy, or rolling a 6 grants an extra turn.

### Game Modes

The environment supports two modes via the `mode` parameter:

* **single** (`mode="single"`, default): Each player competes individually. Players are ranked by the order in which they finish all four pieces.

* **Teams** (`mode="teams"`): Fixed 2v2 team-based play. Teams: (player_0, player_2) vs (player_1, player_3). A team wins when both teammates finish all four pieces. Teammates cannot capture each other but can form team blocks (2+ pieces from the same team on a square). Finished agents can use their dice rolls to move their teammate's pieces (dice-sharing).

### Observation Space

The observation is a **flat numpy array of length 86** (`dtype=np.float32`):

* **Indices 0–69**: Core game state encoding:
  * **0–51**: Main-track occupancy (52 shared squares).
  * **52–67**: Each piece's zone (yard / main / home / finished) and progress across all players.
  * **68**: Normalized current dice value (`dice / 6.0`).
  * **69**: `1.0` if it is this agent's turn, `0.0` otherwise.
* **Indices 70–85**: Full dice bank (up to 16 dice), normalized (`dice / 6.0`). Unused slots are 0.

The action mask is exposed in `info["action_mask"]` as `np.int8` with length matching the action space. Only the currently acting agent has a non-zero action mask. All other agents receive an all-zero mask.

#### Legal Actions Mask

Legal moves for the current agent are given by the action mask in `info["action_mask"]`. The action space is `Discrete(65)` encoding all `(piece_index, dice_index)` pairs plus PASS:

* Actions `0..63`: `(piece_index, dice_index)` where `piece_index ∈ [0,3]` and `dice_index ∈ [0,15]` refers to a die in the agent's dice bank.
* Action `64`: PASS (only legal when no movement actions are available).

In teams mode, if an agent has all pieces finished, their action mask reflects their teammate's legal moves (dice-sharing).

Any action index with mask value 0 is illegal and, when taken, will terminate the episode for the acting agent via `TerminateIllegalWrapper`.

### Action Space

The action space is `Discrete(65)` = `(4 pieces × 16 dice slots) + 1 PASS`. Agents explicitly choose both which piece to move and which die from their bank to use. Dice are banked when rolled (6s grant extra rolls; three consecutive 6s cancel that roll attempt but preserve prior banked dice). Dice remain in the bank across chained extra turns until consumed or the turn ends.

### Rewards

**single mode:**
* Terminal ranks (1st / 2nd / 3rd / 4th): +1.00 / +0.30 / −0.30 / −1.00
* Illegal move: −1 for the acting agent (via wrapper), 0 for others
* All other moves: 0, plus dense shaping (captures, being captured, finishing pieces, leaving the yard, loop waste per full pre-capture loop, and threat exposure that accounts for multiple nearby enemy pieces behind).

**Teams mode:**
* Winning team (both teammates): +1 for each agent on the winning team
* Losing team: −1 for each agent on the losing team
* Illegal move: −1 for the acting agent (via wrapper), 0 for others
* All other moves: 0, plus the same shaping terms as in single (terminal rewards remain ±1 for the two teams).

### Version History

* v0: Initial release.
* v1: Dice banking with explicit die selection. Action space expanded to `Discrete(65)` encoding `(piece, die_index)` pairs. Observation expanded to 86 elements including full dice bank.
"""

from __future__ import annotations

import os
import numpy as np
import pygame
import gymnasium
from gymnasium import spaces
from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
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
    """Ludo environment supporting single-player-free-for-all and 2v2 team modes."""
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "ludo_v0",
        "is_parallelizable": False,
        "render_fps": 4,
    }

    MAIN_TRACK_LEN = 52
    HOME_LEN = 6
    PIECES = 4
    # Maximum number of dice that can be *addressed* by the action space for a
    # single agent's bank. The bank itself is unbounded; if more than this many
    # dice are present, only the first MAX_DICE_SLOTS are selectable.
    MAX_DICE_SLOTS = 16

    # Observation layout (see module docstring for high-level description)
    # Core slice [0, OBS_CORE_LEN):
    #   - [0, OBS_MAIN_LEN): main track occupancy
    #   - [OBS_MAIN_LEN, OBS_MAIN_LEN + OBS_PIECES_LEN): piece zones/progress
    #   - OBS_DICE_IDX: normalized current dice
    #   - OBS_TURN_IDX: turn flag
    # Indices [70, 70 + MAX_DICE_SLOTS) encode the full dice bank (up to
    # MAX_DICE_SLOTS dice), normalized into [0, 1] via division by 6.0.
    OBS_MAIN_LEN = 52
    OBS_PIECES_LEN = 16
    OBS_DICE_IDX = 68
    OBS_TURN_IDX = 69
    OBS_DICE_BANK_START = 70
    OBS_CORE_LEN = 70 + MAX_DICE_SLOTS
    OBS_TOTAL_LEN = OBS_CORE_LEN

    SAFE_SQUARES = {0, 8, 13, 21, 26, 34, 39, 47}
    START_INDEX = [0, 13, 26, 39]
    
    # Reward shaping constants
    REWARD_FINISH_PIECE = [0.10, 0.08, 0.06, 0.04]
    REWARD_LEAVE_YARD = 0.02
    REWARD_LOOP_WASTE = -0.20
    REWARD_CAPTURE_CAP = 0.25
    REWARD_THREAT_MAX = -0.08
    REWARD_THREAT_FACTOR = 0.48
    REWARD_RANK = [1.0, 0.3, -0.3, -1.0]
    
    # Observation encoding constants
    OBS_HOME_BASE = 0.8
    OBS_HOME_DIVISOR = 30.0
    
    # Capture reward thresholds: (threshold, reward_value)
    CAPTURE_REWARD_THRESHOLDS = [(0.2, 0.02), (0.6, 0.03), (0.9, 0.06), (1.0, 0.08)]

    def __init__(self, num_players=4, render_mode=None, screen_scaling=8, mode="single"):
        EzPickle.__init__(self, num_players, render_mode, screen_scaling, mode)
        super().__init__()

        assert 2 <= num_players <= 4
        assert mode in ("single", "teams"), f"mode must be 'single' or 'teams', got '{mode}'"
        if mode == "teams":
            assert num_players == 4, "teams mode requires exactly 4 players"
        
        self.num_players = num_players
        self.render_mode = render_mode
        self.screen_scaling = screen_scaling
        self.mode = mode
        self.team_mode = (mode == "teams")
        
        self.team_map = {f"player_{i}": TEAM_MAP[f"player_{i}"] for i in range(4)} if self.team_mode else {}

        self.agents = [f"player_{i}" for i in range(num_players)]
        self.possible_agents = self.agents[:]
        # Cache agent indices for faster lookups
        self._agent_indices = {a: i for i, a in enumerate(self.agents)}

        # Action encoding:
        #   - For each piece index in [0, PIECES-1] and each dice index in
        #     [0, MAX_DICE_SLOTS-1], we allocate one discrete action.
        #   - One extra action id is reserved for PASS.
        # The agent selects a single integer in [0, PASS_ACTION]; we decode it
        # into (piece_index, dice_index) or PASS internally.
        self.pass_action = self.PIECES * self.MAX_DICE_SLOTS
        self.action_spaces = {
            a: spaces.Discrete(self.pass_action + 1) for a in self.agents
        }
        self.observation_spaces = {
            a: spaces.Box(0.0, 1.0, shape=(self.OBS_TOTAL_LEN,), dtype=np.float32)
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
        # Rebuild agent indices cache
        self._agent_indices = {a: i for i, a in enumerate(self.agents)}
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        # Game end state bookkeeping (used to gate final reward assignment)
        self._game_winner = None  # team_id for teams mode, None for single mode
        self._game_finished = False

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.piece_state = {
            a: [("yard", None) for _ in range(self.PIECES)]
            for a in self.agents
        }

        self.distance = {
            a: [0] * self.PIECES for a in self.agents
        }

        self.current_dice = 0
        self.consecutive_sixes = {a: 0 for a in self.agents}
        # Per-turn dice banking: stores all dice available to each agent for the
        # current turn (including across chained extra turns).
        self.dice_bank = {a: [] for a in self.agents}
        # Track if three sixes were rolled (ends turn immediately)
        self.three_sixes_rolled = {a: False for a in self.agents}

        self.has_captured = {a: False for a in self.agents}

        self.finish_order = []

        # Reward-shaping state
        self.finished_pieces = {a: 0 for a in self.agents}
        self.pre_capture_steps = {
            a: [0] * self.PIECES for a in self.agents
        }
        self.capture_reward_tracker = {
            a: {b: 0.0 for b in self.agents if b != a} for a in self.agents
        }

        self._roll_new_dice()

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------

    def observe(self, agent):
        # Core observation buffer (without action mask). Total length is
        # OBS_TOTAL_LEN; indices [0, OBS_CORE_LEN) are the core state, and the
        # remaining slots are used for auxiliary information such as the dice
        # bank summary.
        obs = np.zeros(self.OBS_TOTAL_LEN, dtype=np.float32)

        # Encode how many pieces occupy each main-track square, normalized into [0, 1].
        # This preserves the distinction between singles and blocks instead of
        # collapsing everything into a single occupancy bit.
        board = np.zeros(self.MAIN_TRACK_LEN, dtype=np.float32)
        for a in self.possible_agents:
            for zone, idx in self.piece_state[a]:
                if zone == "main":
                    board[idx] += 1.0
        if np.any(board):
            # At most `PIECES` pieces of a single colour can share a square. We
            # normalize by PIECES and clip to [0, 1] to remain within the
            # documented observation bounds even if multiple colours stack.
            board = np.clip(board / float(self.PIECES), 0.0, 1.0)
        obs[: self.OBS_MAIN_LEN] = board

        # Observation encoding functions for each zone type
        def encode_yard(idx):
            return 0.0
        def encode_main(idx):
            return (idx + 1) / float(self.MAIN_TRACK_LEN)
        def encode_home(idx):
            return self.OBS_HOME_BASE + idx / self.OBS_HOME_DIVISOR
        def encode_finished(idx):
            return 1.0
        
        encoders = {
            "yard": encode_yard,
            "main": encode_main,
            "home": encode_home,
            "finished": encode_finished
        }
        
        offset = self.OBS_MAIN_LEN
        for a in self.possible_agents:
            for zone, idx in self.piece_state[a]:
                obs[offset] = encoders.get(zone, encode_finished)(idx)
                offset += 1

        # Defensive check: if the layout above changes (e.g., different number of
        # pieces encoded), this assert will fail immediately instead of silently
        # corrupting downstream indices.
        expected_offset = self.OBS_DICE_IDX
        assert (
            offset == expected_offset
        ), f"Observation layout mismatch: expected offset {expected_offset}, got {offset}"

        obs[self.OBS_DICE_IDX] = self.current_dice / 6.0
        obs[self.OBS_TURN_IDX] = 1.0 if agent == self.agent_selection else 0.0

        # Encode the full dice bank (up to MAX_DICE_SLOTS dice) into the
        # observation. Dice values are normalized into [0, 1] via division by
        # 6.0; unused slots (if bank has fewer than MAX_DICE_SLOTS dice) are
        # left at 0.
        bank = self.dice_bank.get(agent, [])
        for i in range(min(self.MAX_DICE_SLOTS, len(bank))):
            obs[self.OBS_DICE_BANK_START + i] = bank[i] / 6.0

        # Full action mask is provided via infos and matches the discrete action
        # space. Only legal (piece, dice_index) pairs and PASS are marked as 1.
        action_n = self.action_spaces[agent].n
        action_mask = np.zeros(action_n, dtype=np.int8)
        if agent == self.agent_selection:
            target = None
            if self.team_mode and self._all_pieces_finished(agent):
                target = self._get_teammate(agent)
            legal = self._legal_actions(agent, target_agent=target) if target or not self.team_mode else [self.pass_action]
            
            for a in legal:
                action_mask[a] = 1

        self.infos[agent]["action_mask"] = action_mask

        return obs

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    # ------------------------------------------------------------
    # Team helpers
    # ------------------------------------------------------------

    def _same_team(self, a, b):
        """True if agents a and b are on the same team (teams mode only)."""
        if not self.team_mode:
            return False
        return self.team_map.get(a) == self.team_map.get(b)

    def _get_teammate(self, agent):
        """Return the teammate of agent in teams mode, else None."""
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
        """True if other is an enemy of agent (different colour or team)."""
        if not self.team_mode:
            return agent != other
        return not self._same_team(agent, other)
    
    def _all_pieces_finished(self, agent):
        """Check if all pieces of an agent are finished."""
        return all(z == "finished" for z, _ in self.piece_state[agent])

    # ------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------

    def _pieces_on_main(self, pos):
        """List of (agent, piece_idx) on a given main-track position."""
        pieces = []
        for a in self.agents:
            for i, (zone, idx) in enumerate(self.piece_state[a]):
                if zone == "main" and idx == pos:
                    pieces.append((a, i))
        return pieces

    def _count_pieces_by_group(self, pieces, filter_func=None):
        """Count pieces grouped by agent (single mode) or team (teams mode).
        
        Args:
            pieces: List of (agent, piece_idx) tuples
            filter_func: Optional function to filter pieces before counting
        
        Returns:
            Dictionary mapping group identifier to count
        """
        if filter_func:
            pieces = [(a, i) for a, i in pieces if filter_func(a)]
        
        if not self.team_mode:
            # Single mode: count per agent/colour
            counts = {}
            for a, _ in pieces:
                counts[a] = counts.get(a, 0) + 1
            return counts
        else:
            # Teams mode: count per team id
            team_counts = {}
            for a, _ in pieces:
                team_id = self.team_map.get(a)
                if team_id is not None:
                    team_counts[team_id] = team_counts.get(team_id, 0) + 1
            return team_counts

    def _is_any_block(self, pos):
        """True if any block (2+ aligned pieces) is on this main-track position."""
        pieces = self._pieces_on_main(pos)
        counts = self._count_pieces_by_group(pieces)
        return any(c >= 2 for c in counts.values())

    def _is_enemy_block(self, agent, pos):
        """True if an enemy block is on this main-track position."""
        pieces = self._pieces_on_main(pos)
        counts = self._count_pieces_by_group(pieces, filter_func=lambda a: self._is_enemy(agent, a))
        return any(c >= 2 for c in counts.values())

    def _is_enemy_occupied(self, agent, pos):
        """True if any enemy piece is on a given main-track position."""
        for a in self.agents:
            if not self._is_enemy(agent, a):
                continue
            for zone, idx in self.piece_state[a]:
                if zone == "main" and idx == pos:
                    return True
        return False

    # ------------------------------------------------------------
    # Progress / reward-shaping helpers
    # ------------------------------------------------------------

    def _progress_to_home(self, agent, piece_idx, zone, idx):
        """Normalized progress-to-home in [0, 1] for a piece."""
        if zone == "yard" or idx is None:
            return 0.0

        if zone in ("home", "finished"):
            return 1.0

        start_idx = self.START_INDEX[self._agent_indices[agent]]
        if self.has_captured[agent]:
            steps = self.distance[agent][piece_idx]
        else:
            steps = (idx - start_idx) % self.MAIN_TRACK_LEN

        denom = max(self.MAIN_TRACK_LEN - 1, 1)
        s = steps / denom
        return float(min(1.0, max(0.0, s)))

    def _get_capture_reward(self, victim_s):
        """Get capture reward value based on victim progress."""
        for threshold, reward in self.CAPTURE_REWARD_THRESHOLDS:
            if victim_s < threshold:
                return reward
        return 0.08  # Default for victim_s >= 1.0

    def _apply_capture_rewards(self, captor, victim, victim_s):
        """Apply capture shaping to captor and victim, capped per pair."""
        base_bonus = self._get_capture_reward(victim_s)

        used = self.capture_reward_tracker.get(captor, {}).get(victim, 0.0)
        remaining = self.REWARD_CAPTURE_CAP - used
        if remaining <= 0.0:
            bonus = 0.0
        else:
            bonus = min(base_bonus, remaining)

        if bonus > 0.0:
            self._grant_reward(captor, bonus)
            self.capture_reward_tracker[captor][victim] = used + bonus

        # Defer penalty for victim to apply when they act
        penalty = -self._get_capture_reward(victim_s)
        # Apply victim penalty immediately into per-step rewards; it will be
        # surfaced to the victim on their next `last()` call via
        # `_cumulative_rewards` (PettingZoo AEC semantics).
        self._grant_reward(victim, penalty)

    def _apply_finish_piece_reward(self, mover, owner):
        """Shaping for finishing pieces with diminishing rewards."""
        count = self.finished_pieces[owner] + 1
        self.finished_pieces[owner] = count
        idx = min(count - 1, len(self.REWARD_FINISH_PIECE) - 1)
        self._grant_reward(mover, self.REWARD_FINISH_PIECE[idx])

    def _grant_reward(self, recipient, amount):
        """Grant a shaping or terminal reward to `recipient` for the current step.

        Rewards are written into ``self.rewards`` and then accumulated into
        ``self._cumulative_rewards`` via ``_accumulate_rewards()`` at the end of
        :meth:`step`. This matches the standard PettingZoo AEC pattern where
        :meth:`last` returns the cumulative reward gathered since the agent last
        acted, and the tests reconstruct that value from ``env.rewards``.
        """
        if amount == 0.0:
            return
        if recipient not in self.rewards:
            return
        self.rewards[recipient] += float(amount)

    def _enemy_agents_before_next_turn(self, agent):
        """All enemy agents in turn order who may act before `agent` gets another turn."""
        n = len(self.agents)
        if n <= 1:
            return []

        start = self._agent_indices[agent]
        enemies = []
        for offset in range(1, n):
            idx = (start + offset) % n
            other = self.agents[idx]
            if self.terminations.get(other, False) or self.truncations.get(other, False):
                continue
            if self._is_enemy(agent, other) and other not in enemies:
                enemies.append(other)
        return enemies

    def _apply_threat_penalty(self, mover, owner, piece_idx):
        """Penalty if the moved piece is in near-term capture threat."""
        zone, idx = self.piece_state[owner][piece_idx]
        if zone != "main" or idx in self.SAFE_SQUARES:
            return

        enemy_agents = self._enemy_agents_before_next_turn(mover)
        if not enemy_agents:
            return

        # Aggregate capture probability from *all* enemy pieces behind within 6+6+5 squares.
        # Distance calculation accounts for wrap-around: we want the clockwise distance
        # from enemy piece to our piece (enemy is behind if distance is 1-17 squares).
        p_total = 0.0
        for enemy in enemy_agents:
            for z_e, pos_e in self.piece_state[enemy]:
                if z_e != "main":
                    continue
                # Calculate clockwise distance: (our_pos - enemy_pos) mod track_length
                # This gives distance enemy needs to travel to reach our position
                d = (idx - pos_e) % self.MAIN_TRACK_LEN
                if d == 0:
                    continue  # Same position, skip
                if 1 <= d <= 6:
                    if idx in self.SAFE_SQUARES:
                        continue
                    if self._is_any_block(idx):
                        continue
                    p_total += 1.0 / 6.0
                elif 7 <= d <= 12:
                    p_total += 1.0 / 36.0
                elif 13 <= d <= 17:
                    p_total += 1.0 / 216.0

        if p_total <= 0.0:
            return

        s = self._progress_to_home(owner, piece_idx, zone, idx)
        penalty = -self.REWARD_THREAT_FACTOR * p_total * (0.5 + 0.5 * s)
        # Clamp to [REWARD_THREAT_MAX, 0].
        penalty = max(self.REWARD_THREAT_MAX, min(0.0, penalty))
        self._grant_reward(mover, penalty)

    def _legal_pieces_with_die(self, check_agent, die_value):
        """Return legal piece indices for `check_agent` given a specific die value."""
        legal_pieces = []
        for i, (zone, idx) in enumerate(self.piece_state[check_agent]):
            if zone == "finished":
                continue

            if zone == "yard" and die_value == 6:
                start = self.START_INDEX[self._agent_indices[check_agent]]
                if start in self.SAFE_SQUARES or not self._is_any_block(start):
                    legal_pieces.append(i)

            elif zone == "main":
                if not self.has_captured[check_agent]:
                    blocked = False
                    for step in range(1, die_value + 1):
                        pos = (idx + step) % self.MAIN_TRACK_LEN
                        if pos in self.SAFE_SQUARES:
                            continue
                        if self._is_any_block(pos):
                            blocked = True
                            break
                    if not blocked:
                        legal_pieces.append(i)
                else:
                    # After the first capture for this colour, pieces advance towards
                    # their home track using a distance measure. We allow moves up to
                    # and including those that land exactly on the final home square,
                    # but not beyond it.
                    new_dist = self.distance[check_agent][i] + die_value
                    max_dist = (self.MAIN_TRACK_LEN - 2) + self.HOME_LEN
                    if new_dist <= max_dist:
                        blocked = False
                        for step in range(1, die_value + 1):
                            dist = self.distance[check_agent][i] + step
                            if dist > self.MAIN_TRACK_LEN - 2:
                                # Entering home track - check if home entry is blocked
                                # Home entry is at the square just before the home track
                                entry_pos = (idx + step) % self.MAIN_TRACK_LEN
                                if entry_pos not in self.SAFE_SQUARES and self._is_any_block(entry_pos):
                                    blocked = True
                                    break
                                break
                            pos = (idx + step) % self.MAIN_TRACK_LEN
                            if pos in self.SAFE_SQUARES:
                                continue
                            if self._is_any_block(pos):
                                blocked = True
                                break
                        if not blocked:
                            legal_pieces.append(i)

            elif zone == "home":
                # Legal if the move stays within home track (idx + die_value <= HOME_LEN - 1)
                # or finishes the piece (idx + die_value == HOME_LEN - 1)
                if idx + die_value <= self.HOME_LEN - 1:
                    legal_pieces.append(i)

        return legal_pieces

    def _encode_action(self, piece_idx, die_idx):
        """Encode (piece_idx, die_idx) into a discrete action id."""
        return piece_idx * self.MAX_DICE_SLOTS + die_idx

    def _decode_action(self, action):
        """Decode a discrete action id into ('pass', None, None) or ('move', piece, die_idx)."""
        if action == self.pass_action:
            return "pass", None, None
        piece_idx = action // self.MAX_DICE_SLOTS
        die_idx = action % self.MAX_DICE_SLOTS
        return "move", piece_idx, die_idx

    def _legal_actions(self, agent, target_agent=None):
        """Return legal discrete actions for the current agent."""
        check_agent = target_agent if target_agent is not None else agent

        bank = self.dice_bank.get(agent, [])
        legal_actions = []

        # For each die in the bank (up to MAX_DICE_SLOTS), compute which pieces
        # are legal to move with that die and encode (piece, die_index) pairs.
        for die_idx, die_value in enumerate(bank[: self.MAX_DICE_SLOTS]):
            legal_pieces = self._legal_pieces_with_die(check_agent, die_value)
            for piece_idx in legal_pieces:
                # Guard against malformed piece indices.
                if 0 <= piece_idx < self.PIECES:
                    legal_actions.append(self._encode_action(piece_idx, die_idx))

        # If no (piece, die) pair is legal, only PASS is legal.
        if not legal_actions:
            return [self.pass_action]

        return legal_actions

    # ------------------------------------------------------------
    # Dice handling
    # ------------------------------------------------------------

    def _roll_new_dice(self):
        """Roll dice for the current agent and bank them with a three-sixes penalty.

        Dice banking rules:
          * Rolling a 6 does NOT immediately move a piece; it is added to the
            agent's dice bank and the agent rolls again.
          * Rolling a non-6 ends the roll attempt and is added to the bank.
          * Dice bank persists across chained extra turns.

        Three-sixes penalty (per roll attempt):
          * If the agent rolls three consecutive 6s in this *single* roll
            attempt, all dice from this attempt are discarded, but any dice
            already in the bank from earlier in the turn are preserved.
          * The turn ends immediately (skipped).
        """
        agent = self.agent_selection
        attempt_dice = []
        six_count = 0
        self.three_sixes_rolled[agent] = False

        while True:
            roll = int(self.np_random.integers(1, 7))
            if roll == 6:
                six_count += 1
                if six_count >= 3:
                    # Three-sixes penalty: discard this attempt entirely and end turn.
                    attempt_dice = []
                    self.three_sixes_rolled[agent] = True
                    break
                attempt_dice.append(6)
            else:
                attempt_dice.append(roll)
                break

        if attempt_dice:
            self.dice_bank[agent].extend(attempt_dice)
            # Limit bank size to MAX_DICE_SLOTS to prevent unbounded growth
            if len(self.dice_bank[agent]) > self.MAX_DICE_SLOTS:
                self.dice_bank[agent] = self.dice_bank[agent][:self.MAX_DICE_SLOTS]

        # Expose one die to the observation / legal move logic. For now we use
        # the first die in the bank; remaining dice stay available in the bank.
        if self.dice_bank[agent]:
            self.current_dice = self.dice_bank[agent][0]
        else:
            self.current_dice = 0

    # ------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------

    def _end_turn(self, agent, accumulate=True):
        """End the current turn: clear dice bank, move to next agent, roll new dice."""
        self.dice_bank[agent] = []
        self.agent_selection = self._agent_selector.next()
        self._roll_new_dice()
        if accumulate:
            self._accumulate_rewards()

    def _validate_action_params(self, agent, die_idx, piece_idx, bank):
        """Validate action parameters. Returns True if valid, False otherwise."""
        if not (0 <= die_idx < len(bank)):
            return False
        if not (0 <= piece_idx < self.PIECES):
            return False
        return True

    def _assign_rank_rewards(self):
        """Assign terminal rank rewards to all agents based on finish order."""
        self._game_finished = True
        for idx, a in enumerate(self.finish_order):
            r = self.REWARD_RANK[idx] if idx < len(self.REWARD_RANK) else self.REWARD_RANK[-1]
            self._grant_reward(a, r)
            self.terminations[a] = True

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

        # PettingZoo AEC pattern: start each live-agent step by clearing the
        # per-step rewards dict and zeroing the acting agent's cumulative
        # reward. Shaping rewards for non-acting agents are already preserved
        # in ``_cumulative_rewards`` and are not affected by this.
        self._clear_rewards()
        self._cumulative_rewards[agent] = 0.0

        target_agent = agent
        mode, action_piece_idx, die_idx = self._decode_action(action)
        all_finished = False
        if self.team_mode:
            all_finished = self._all_pieces_finished(agent)
            if all_finished:
                teammate = self._get_teammate(agent)
                if teammate is not None and not (self.terminations.get(teammate, False) or self.truncations.get(teammate, False)):
                    # Finished agents reuse their dice to move a teammate's
                    # pieces; we keep the decoded `action_piece_idx` and only
                    # retarget which agent's piece_state we index into.
                    target_agent = teammate
                else:
                    # Teammate is terminated or doesn't exist, can only pass
                    if mode != "pass":
                        # Treat as illegal action
                        self._end_turn(agent)
                        return

        legal = self._legal_actions(
            agent,
            target_agent=target_agent if self.team_mode and all_finished else None,
        )

        # TerminateIllegalWrapper ensures that only legal actions reach this
        # point. The guard is kept for robustness but should normally be dead
        # code.
        if action not in legal:
            # Illegal action ends the turn and clears any remaining banked dice.
            self._end_turn(agent)
            return

        capture = False
        finished_this_move = False

        if mode == "pass":
            # PASS: end the turn, clear any remaining banked dice for this agent.
            self._end_turn(agent, accumulate=False)
            return
        
        # Consume the explicitly chosen die from the acting agent's dice bank.
        bank = self.dice_bank[agent]
        if not self._validate_action_params(agent, die_idx, action_piece_idx, bank):
            # Invalid parameters: abort move, end turn, clear bank
            self._end_turn(agent)
            return
        
        die = bank.pop(die_idx)
        self.current_dice = die

        zone, idx = self.piece_state[target_agent][action_piece_idx]

        if zone == "yard":
            start = self.START_INDEX[self._agent_indices[target_agent]]

            active_pieces = sum(
                1 for z, _ in self.piece_state[target_agent] if z in ("main", "home")
            )
            forced = len(legal) == 1
            if active_pieces >= 1 and not forced:
                self._grant_reward(agent, self.REWARD_LEAVE_YARD)

            self.piece_state[target_agent][action_piece_idx] = ("main", start)
            self.distance[target_agent][action_piece_idx] = 0
            self.pre_capture_steps[target_agent][action_piece_idx] = 0

        elif zone == "main":
            roll = self.current_dice
            if not self.has_captured[target_agent]:
                old_steps = self.pre_capture_steps[target_agent][action_piece_idx]
                new_steps = old_steps + roll
                loops_before = old_steps // self.MAIN_TRACK_LEN
                loops_after = new_steps // self.MAIN_TRACK_LEN
                wasted_loops = max(0, loops_after - loops_before)
                if wasted_loops > 0:
                    self._grant_reward(agent, self.REWARD_LOOP_WASTE * wasted_loops)
                self.pre_capture_steps[target_agent][action_piece_idx] = new_steps

                new_pos = (idx + roll) % self.MAIN_TRACK_LEN
                self.piece_state[target_agent][action_piece_idx] = ("main", new_pos)
                capture = self._check_capture(target_agent, new_pos)
            else:
                current_dist = self.distance[target_agent][action_piece_idx]
                new_dist = current_dist + roll
                self.distance[target_agent][action_piece_idx] = new_dist

                last_main_distance = self.MAIN_TRACK_LEN - 2
                if new_dist <= last_main_distance:
                    new_pos = (idx + roll) % self.MAIN_TRACK_LEN
                    self.piece_state[target_agent][action_piece_idx] = ("main", new_pos)
                    capture = self._check_capture(target_agent, new_pos)
                else:
                    steps_to_entry = last_main_distance - current_dist
                    steps_in_home = roll - steps_to_entry - 1
                    # Ensure steps_in_home is non-negative
                    if steps_in_home < 0:
                        # This shouldn't happen if legal moves are computed correctly,
                        # but handle gracefully by finishing the piece
                        steps_in_home = self.HOME_LEN - 1
                    if steps_in_home >= self.HOME_LEN - 1:
                        self.piece_state[target_agent][action_piece_idx] = ("finished", None)
                        finished_this_move = True
                    else:
                        # Otherwise, land somewhere on the home track.
                        self.piece_state[target_agent][action_piece_idx] = ("home", steps_in_home)

        elif zone == "home":
            new_idx = idx + self.current_dice
            if new_idx >= self.HOME_LEN - 1:
                # Finish piece if we reach or exceed the final home square
                self.piece_state[target_agent][action_piece_idx] = ("finished", None)
                finished_this_move = True
            else:
                self.piece_state[target_agent][action_piece_idx] = ("home", new_idx)

        if finished_this_move:
            self._apply_finish_piece_reward(agent, target_agent)

        # Win check - when the game ends, assign terminal rewards immediately to
        # all agents and mark them terminated. Rewards are accumulated below.
        if self.team_mode:
            team_id = self.team_map.get(target_agent)
            if team_id is not None:
                team_agents = [a for a in self.agents if self.team_map.get(a) == team_id]
                team_finished = all(self._all_pieces_finished(a) for a in team_agents)
                if team_finished and not self._game_finished:
                    self._game_winner = team_id
                    self._game_finished = True
                    for a in self.agents:
                        if self.team_map.get(a) == team_id:
                            self._grant_reward(a, 1.0)
                        else:
                            self._grant_reward(a, -1.0)
                        self.terminations[a] = True
        else:
            if self._all_pieces_finished(agent):
                if agent not in self.finish_order:
                    self.finish_order.append(agent)

                # Check if game should end: when all players have finished
                # or when num_players - 1 have finished (last player is determined)
                if not self._game_finished:
                    finished_count = len(self.finish_order)
                    if finished_count >= self.num_players - 1:
                        # If exactly num_players - 1 finished, add remaining player
                        if finished_count == self.num_players - 1:
                            remaining = [a for a in self.agents if a not in self.finish_order]
                            if remaining:
                                self.finish_order.extend(remaining)
                        self._assign_rank_rewards()

        self._apply_threat_penalty(agent, target_agent, action_piece_idx)

        # Extra turn is granted for:
        # - Capturing an enemy piece
        # - Finishing a piece
        # Note: Rolling a 6 grants an extra roll (handled in _roll_new_dice), but using
        # a banked 6 does NOT grant an extra turn - that would be double-counting.
        extra_turn = capture or finished_this_move

        if self.render_mode == "human":
            self.render()

        if not self._game_finished:
            # Check for three-sixes penalty: if three sixes were rolled, end turn immediately
            if self.three_sixes_rolled.get(agent, False):
                # Three-sixes penalty: turn ends immediately, clear bank
                self.three_sixes_rolled[agent] = False
                self._end_turn(agent, accumulate=False)
            elif not extra_turn:
                # Turn ends normally: clear any remaining banked dice before passing play.
                self._end_turn(agent, accumulate=False)
            else:
                # Extra turn: roll more dice into the existing bank for this agent.
                self._roll_new_dice()

        # Finally, roll per-step rewards into the cumulative buffer consumed by
        # :meth:`last`, so that the compliance tests' reconstructed rewards
        # match exactly.
        self._accumulate_rewards()

    # ------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------

    def _check_capture(self, agent, pos):
        if pos in self.SAFE_SQUARES:
            return False

        if self._is_any_block(pos):
            return False

        pieces = self._pieces_on_main(pos)
        counts = {}
        indices = {}
        for a, i in pieces:
            if not self._is_enemy(agent, a):
                continue
            counts[a] = counts.get(a, 0) + 1
            indices.setdefault(a, []).append(i)

        if not counts:
            return False

        for a, c in counts.items():
            if c == 1:
                i = indices[a][0]

                victim_zone = "main"
                victim_idx = pos
                s_victim = self._progress_to_home(a, i, victim_zone, victim_idx)
                self._apply_capture_rewards(agent, a, s_victim)

                # Send the captured piece back to yard and reset its progress
                # bookkeeping completely.
                self.piece_state[a][i] = ("yard", None)
                self.distance[a][i] = 0
                self.pre_capture_steps[a][i] = 0

                # Mark that this colour has captured at least one enemy piece.
                self.has_captured[agent] = True

                # After the first capture for this agent, recompute both
                # distance and pre_capture_steps for all of its pieces so that
                # the two progress trackers remain consistent and home-entry
                # behaviour matches the standard rules.
                start_idx = self.START_INDEX[self._agent_indices[agent]]
                for p_idx, (z, idx2) in enumerate(self.piece_state[agent]):
                    if z == "main" and idx2 is not None:
                        steps = (idx2 - start_idx) % self.MAIN_TRACK_LEN
                        self.distance[agent][p_idx] = steps
                        self.pre_capture_steps[agent][p_idx] = steps
                    elif z in ("home", "finished"):
                        # Preserve distance for pieces already in home/finished
                        # They've already passed the main track entry point
                        continue
                    else:
                        # For yard pieces, reset both trackers to zero
                        self.distance[agent][p_idx] = 0
                        self.pre_capture_steps[agent][p_idx] = 0
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
            if self.has_captured.get(agent, False):
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
