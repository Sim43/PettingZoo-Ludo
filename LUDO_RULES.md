## Ludo Rules in This Environment

This document describes the game rules as implemented in the PettingZoo-Ludo environment, including the optional 2v2 team mode and the capture-before-home extension.

---

## Board and Piece Movement

- **Players / colours**: up to four – `player_0` (Green), `player_1` (Yellow), `player_2` (Blue), `player_3` (Red).
- **Pieces**: each player has 4 pieces that start in their **yard**.
- **Tracks**:
  - **Main track**: 52 shared squares (indices 0–51).
  - **Home track**: 6 colour-specific squares per player (indices 0–5 in code, 0–4 as intermediate, 5 as finished).

### Yard entry

- A piece leaves the yard only on a **roll of 6**, entering the main track at a colour-specific start index:
  - Green (`player_0`) → main index **0**
  - Yellow (`player_1`) → main index **13**
  - Blue (`player_2`) → main index **26**
  - Red (`player_3`) → main index **39**

### Last main square per colour

Movement on the main track is tracked as a colour-specific distance from the start. The last main square each colour can occupy before entering home is:

- Green: main index **50**
- Yellow: main index **11**
- Blue: main index **24**
- Red: main index **37**

After the capture requirement is satisfied (see below), any move that goes beyond this last main-square distance enters the corresponding **home track** in the same move.

---

## Safe Squares, Blocks, and Captures

### Safe squares

- Certain main-track indices are **safe squares** where captures are **never allowed**.
- On safe squares:
  - Pieces of **any colours** may stack freely.
  - Passing through or landing on a safe square is **never blocked**, even by blocks.

### Blocks

- A **block** is 2 or more aligned pieces that cannot be captured and that block passage on **non-safe** squares:
  - **Free-for-all** (`mode="ffa"`):
    - A block is **two or more pieces of the same colour** on a main square.
  - **Teams mode** (`mode="teams"`):
    - A block is **two or more pieces from the same team** on a main square, possibly split across teammates.
    - Teammates can form **team blocks**.
- On **non-safe** squares:
  - Blocks cannot be captured.
  - Moves that would pass through or land on a blocked square are illegal.

### Captures

- On **non-safe**, non-blocked main-track squares:
  - If there is **exactly one enemy piece** on that square, moving onto it **captures** that piece and sends it back to its yard.
  - Squares with 0 or 2+ enemy pieces (i.e., blocks) are not capturable.

---

## Capture Requirement for Home Entry

Each colour must **individually** capture at least one enemy piece before any of its own pieces may enter the home track.

- A per-colour flag is tracked internally:
  - `has_captured[agent] = False` initially for every colour.
  - When `agent` captures any enemy piece once, `has_captured[agent]` is set to `True` and never reset for the rest of the game.
- This rule applies in both **FFA** and **teams** mode:
  - In teams mode, captures are **not shared** between teammates.
  - A capture by `player_0` does **not** unlock home entry for `player_2`, and vice versa.

### Pre-capture behaviour (has_captured[colour] == False)

Until a colour has captured at least one enemy piece:

- Its home-entry square behaves like a **normal main-track square**.
- Any move that would normally enter home instead:
  - **Stays entirely on the main track**, using modulo arithmetic to wrap around.
  - The piece moves `dice` steps along the main loop, even if that passes the usual home-entry point.
- Only standard blocking rules apply:
  - Safe squares always allow passage/stacking.
  - Non-safe blocked squares still prevent passage.

**Example:**

- A piece is 3 squares before its colour’s home entry.
- That colour has `has_captured == False`.
- Dice roll = 5.
- The piece moves 5 squares on the main track, wrapping past the entry, and does **not** enter home.

### Post-capture behaviour (has_captured[colour] == True)

Once a colour has captured at least one enemy piece:

- Its home-entry square becomes a **mandatory gate into the home track**.
- Wrapping past the home entry on the main track is **no longer allowed**.
- Any move whose total distance from that colour’s start:
  - Is **≤ last-main distance**: remains on the main track.
  - **Exceeds** the last-main distance: consumes the remaining steps on main and uses the rest to move along the home track.
    - If the remaining steps in home reach or pass the final home index, the piece is immediately marked **finished**.

This capture-unlock behaviour is **irreversible** per colour.

---

## Three Sixes Rule

- For each colour, the environment tracks the number of **consecutive rolls of 6**.
- If a player rolls three 6s in a row:
  - Their turn is **skipped**.
  - The dice is reset to 0 for that player.
  - The turn immediately advances to the next agent, who then rolls.

This rule applies in both FFA and teams mode.

---

## Game Modes and Team-Specific Rules

### Free-for-all (`mode="ffa"`)

- Each colour is an **independent agent**.
- Win condition:
  - The game ends as soon as any agent has all 4 pieces in the final home position.
  - That agent receives **+1**, all others receive **-1**.
- Capture requirement (above) applies per colour.
- Blocks, captures, and safe squares behave as defined in earlier sections.

### Teams (`mode="teams"`)

- Teams are fixed:
  - **Team 0**: `player_0` (Green) and `player_2` (Blue)
  - **Team 1**: `player_1` (Yellow) and `player_3` (Red)

#### Team win condition

- A team wins when **all pieces of both teammates** reach the final home position (`finished`).
- At team win:
  - Agents on the winning team receive **+1**.
  - Agents on the losing team receive **-1**.
  - The episode terminates for all agents.

#### Team blocks

- Blocks are per-team in this mode:
  - Any combination of 2 or more pieces from the same team on a main-square form a block.
  - Team blocks cannot be captured and block passage for both teams on non-safe squares.

#### Dice-sharing (helping teammate)

- If a player has all 4 of their own pieces **finished**, they **still take turns**.
- On their turn:
  - If they have no movable own pieces, their legal moves are computed from their **teammate’s** movable pieces.
  - They use their dice roll to move a teammate’s piece, obeying the same movement, blocking, capturing, and capture-requirement rules.
- The action mask and action indices remain:
  - `0–3` → possible piece indices (now referring to teammate’s pieces when helping).
  - `4` → PASS (only legal if no movement is possible).

The capture requirement is always per colour, **not per team**, even in dice-sharing situations.

---

## Visual Capture Markers

- For each colour, there is a precomputed center of its 4 yard coordinates.
- Once `has_captured[colour]` becomes `True`, the renderer draws a **small coloured circle** at that yard-center position:
  - Acts as a permanent indicator that this colour has captured at least one enemy piece.
  - Persists for the rest of the episode.

This marker is cosmetic only and does not affect game logic or observations.

