## PettingZoo Ludo Environment

**PettingZoo-Ludo** is a third-party, turn-based multi-agent environment that implements the board game **Ludo** using the PettingZoo AEC API and Gymnasium spaces. It is designed as a standalone package that can be listed in PettingZoo's third-party registry.

![Ludo demo](assets/demo.gif)

---

## Installation

- **Clone and install in editable mode**:

```bash
git clone https://github.com/Sim43/PettingZoo-Ludo.git
cd PettingZoo-Ludo
pip install -e .
```

- **Runtime dependencies** (also listed in `setup.py`):
  - **Python**: 3.8+
  - **PettingZoo** `>=1.24.0`
  - **Gymnasium** `>=1.0.0`
  - **NumPy** `>=1.21.0`
  - **pygame** `>=2.1.0`

Testing and tooling dependencies include `pytest`, `jinja2`, `typeguard`, `lark`, and `setuptools`.

---

## Quickstart

**Basic usage (no rendering, free-for-all by default):**

```python
from ludo.ludo import env

# Free-for-all mode (default)
env = env()
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        # Use action mask to select a legal action
        mask = info.get("action_mask", None)
        if mask is not None:
            legal_actions = [i for i, v in enumerate(mask) if v == 1]
            action = legal_actions[0] if legal_actions else 0
        else:
            action = env.action_space(agent).sample()
    env.step(action)
```

**With human rendering (and optional team mode):**

```python
from ludo.ludo import env

# Free-for-all
env = env(render_mode="human")
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        mask = info.get("action_mask", None)
        legal = [i for i, v in enumerate(mask) if v == 1] if mask is not None else []
        action = legal[0] if legal else 0
    env.step(action)
env.close()

# 2v2 teams: (player_0, player_2) vs (player_1, player_3)
env = env(render_mode="human", mode="teams")
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        mask = info.get("action_mask", None)
        legal = [i for i, v in enumerate(mask) if v == 1] if mask is not None else []
        action = legal[0] if legal else 0
    env.step(action)
env.close()
```

---

## Environment details

### Agents, turns, and game modes

- **Agents**: up to **4 players** (`"player_0"`–`"player_3"`), configurable via `num_players` in `raw_env.__init__`.
- **Turn order**: sequential, managed with PettingZoo's `agent_selector`.
- **Three sixes rule**: three consecutive rolls of 6 for the same player cause their turn to be skipped and the dice reset.
-- **Game modes**:
  - **Free-for-all** (`mode="ffa"`, default): each player is an independent agent competing to finish all of their own pieces first.
  - **Teams** (`mode="teams"`): fixed 2v2 teams:
    - Team 0: `player_0` (Green) and `player_2` (Blue)
    - Team 1: `player_1` (Yellow) and `player_3` (Red)
    - When all pieces of both teammates are finished, the team wins and the episode terminates.
    - In teams mode, finished agents still take turns and may use their dice rolls to move **their teammate's** pieces (dice-sharing).

### Action space

- **Per-agent action space**: `gymnasium.spaces.Discrete(5)`
  - **0–3**: move the corresponding piece index (0–3).
  - **4**: **PASS**, only legal when no movement actions are available.
- Illegal actions are handled by `TerminateIllegalWrapper`, which uses an **action mask** to determine legal actions.

### Observation space

- **Per-agent observation space**: `gymnasium.spaces.Box(0.0, 1.0, shape=(80,), dtype=np.float32)`
  - A **flat vector of length 80**:
    - **First 75 entries**: core board and game-state features.
    - **Last 5 entries**: binary **action mask** for actions 0–4.
- Key encoding details (see `ludo/ludo.py` for exact logic):
  - **Indices 0–51**: main-track occupancy (shared 52 squares).
  - **Indices 52–67**: piece zones and progress (yard / main / home / finished) across all players.
  - **Index 68**: normalized dice value (`dice / 6.0`).
  - **Index 69**: `1.0` if it is this agent's turn, `0.0` otherwise.
  - **Indices 75–79**: action mask (1.0 = legal, 0.0 = illegal) exposed in the observation and as `info["action_mask"]` with `dtype=np.int8` for sampling.

### Rewards and termination

- **Rewards (free-for-all)**:
  - **+1** for the winning agent (all 4 pieces finished).
  - **-1** for each losing agent at game end.
  - **-1** for an illegal move (via `TerminateIllegalWrapper`) for the acting agent.
  - **0** for all other intermediate moves.
- **Rewards (teams)**:
  - **+1** for each agent on the winning team (both teammates have all 4 pieces finished).
  - **-1** for each agent on the losing team.
  - **-1** for an illegal move (via `TerminateIllegalWrapper`) for the acting agent.
  - **0** for all other intermediate moves.
- **Terminations**:
  - **Free-for-all**: episode ends when any player gets all four pieces to the final home position.
  - **Teams**: episode ends when all pieces of both teammates on a team are finished.
  - In both modes, per-agent `terminations[agent]` flags are used, as required by the AEC API.
- **Truncations**:
  - Currently no built-in `max_cycles`; `truncations` remain `False` unless integrated into a higher-level wrapper.

### Ludo-specific movement rules

- **Starting from the yard**:
  - A piece leaves the yard only on a roll of 6, entering the shared main track at a color-specific start index:
    - **Green (`player_0`)**: main index **0**
    - **Yellow (`player_1`)**: main index **13**
    - **Blue (`player_2`)**: main index **26**
    - **Red (`player_3`)**: main index **39**
- **Last main-track square before home** (per color, encoded via distance from the color's start):
  - **Green**: index **50**
  - **Yellow**: index **11**
  - **Blue**: index **24**
  - **Red**: index **37**
  - Any move that would go beyond this last main square automatically continues into the **home track in the same move**, so pieces never skip home entry or remain on the main track after passing it.
- **Safe squares and blocks**:
  - Certain main-track indices are **safe squares** where captures are not allowed.
  - **Blocks** (two or more pieces of the same player on a main square) cannot be captured and also block passage for other pieces.
- **Captures**:
  - Non-safe, non-blocked squares with exactly one opponent piece allow captures, sending that piece back to its yard.

---

## Rendering

- **Render modes** (in `raw_env.metadata`):
  - `"human"`: Pygame window with a graphical board and pieces.
  - `"rgb_array"`: returns an `H x W x 3` NumPy array of the current frame.
- **Assets**:
  - Board and piece sprites are located in `ludo/img/` and loaded via `pygame`.
  - `assets/demo.gif` shows an example playthrough.
- **Notes**:
  - Rendering requires a display-capable environment when using `"human"`.
  - The environment scales the board and pieces to a fixed window size (~800x800).

---

## Testing and development

- **API compliance and regression tests** are provided under `tests/`:
  - `tests/test_ludo_compliance.py`:
    - PettingZoo **API compliance test** (`api_test`).
    - **Seed test** (`seed_test`) for determinism.
    - Optional performance benchmark and basic save-observation loop.
  - `tests/test_render.py`:
    - Simple script to run the environment with `"human"` rendering and random legal actions.

- **Run tests locally**:

```bash
pytest -v
```

---

## License

- This project is licensed under the terms of the license in `LICENSE` (currently **MIT** unless changed). Please review that file for full details.