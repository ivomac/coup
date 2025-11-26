# Coup

This environment is a [PettingZoo](https://pettingzoo.farama.org/) implementation of the card game [Coup](https://en.wikipedia.org/wiki/Coup_(card_game)).

| Import               | `from pettingzoo_coup import coup_v0`      |
|----------------------|--------------------------------------------|
| Actions              | Discrete                                   |
| Parallel API         | No                                         |
| Manual Control       | No                                         |
| Agents               | `agents=['agent_0', 'agent_1', ...]`       |
| Agents               | 2+ (configurable)                          |
| Action Shape         | varies, default is (31,)                   |
| Action Values        | varies                                     |
| Observation Shape    | varies, default is (94,)                   |
| Observation Values   | varies                                     |

Coup is a 2-6 player competitive hidden information game where the last player standing wins.
Each player starts with 2 cards and 2 coins.
On their turn, players can perform various actions to gain coins, steal from others, or force them to lose cards.
Players can claim to have specific character cards to perform special actions, but opponents can challenge these claims.
If caught bluffing, the claimant loses a card of their choice. If they have the card, the challenger loses a card instead.
Some of the card actions are counter-actions that block and nullify certain actions. These block actions can also be challenged.

See [examples/run.py](./examples/run.py) for basic usage.

### Environment arguments

The environment is configurable to allow different initial conditions of the game. In particular, it is possible to start in a late-game situation
with cards already eliminated and a few players remaining. This way a curriculum-based learning is possible.

``` python
coup_v0.env(
    deck=None, dead_draw=False, num_players=6, num_players_alive=6, render_mode="ansi"
)
```

`deck`: Dictionary mapping card names to their count in the deck. The default deck is 3 of each card type (Ambassador, Assassin, Captain, Contessa, Duke).

`dead_draw`: Whether to draw cards for non-participating players at game start to maintain consistent deck size.

`num_players`: Total number of player slots (2-6). This influences the observation and action space sizes.

`num_players_alive`: Number of players actually participating in the game (must be <= num_players and >= 2).

`render_mode`: Rendering mode. Currently only supports "ansi" for text-based rendering.

### Observation Space

The observation is a 1D numpy array with information about the game state and the current turn's history, from the relative perspective of the observing agent.
The size and range of the observation space depends on initialization parameters (deck and num_players). For default settings (six players, default deck):

|  Index Range  |  Array Length     |  Array Type  |  Description             |  Value Range  |  Scope    |
|---------------|-------------------|--------------|--------------------------|---------------|-----------|
|  0 - 5        |  # Players        |  Counts      |  Coins of each player    |  0 - 12       |  Public   |
|  6 - 10       |  # Card types     |  Counts      |  Unseen card totals      |  0 - 3        |  Private  |
|  11 - 15      |  # Card types     |  Counts      |  Player's cards by type  |  0 - 3        |  Private  |
|  16 - 21      |  # Players        |  Counts      |  Card totals per player  |  0 - 4        |  Public   |
|  22 - 28      |  # Start Actions  |  One-hot     |  Action                  |  0 - 1        |  Public   |
|  29 - 34      |  # Players        |  One-hot     |  Actor                   |  0 - 1        |  Public   |
|  35 - 40      |  # Players        |  One-hot     |  Action's target         |  0 - 1        |  Public   |
|  41 - 46      |  # Players        |  Indicator   |  Challenge passed        |  0 - 1        |  Public   |
|  47 - 52      |  # Players        |  One-hot     |  Challenger              |  0 - 1        |  Public   |
|  53 - 58      |  # Players        |  One-hot     |  Challenge loser         |  0 - 1        |  Public   |
|  59 - 63      |  # Block Actions  |  One-hot     |  Block action            |  0 - 1        |  Public   |
|  64 - 69      |  # Players        |  Indicator   |  Block passed            |  0 - 1        |  Public   |
|  70 - 75      |  # Players        |  One-hot     |  Blocker                 |  0 - 1        |  Public   |
|  76 - 81      |  # Players        |  Indicator   |  Block challenge passed  |  0 - 1        |  Public   |
|  82 - 87      |  # Players        |  One-hot     |  Block challenger        |  0 - 1        |  Public   |
|  88 - 93      |  # Players        |  One-hot     |  Block challenge loser   |  0 - 1        |  Public   |

The subarrays of the observation array of size equal to the number of players are represented relative to the observing agent:
The observer is always at the 0-bit of the subarray, followed by the other players in order of play after the observer.

#### Observation History

Past turn observations are made available as a list of observation arrays in `info["observation_history"]`, from oldest to most recent.

The current observation array only contains information on the current game state and turn. Being a hidden information game, past turns contain important
information that the agent might use to make better decisions. Additionally, not all agents have a chance to observe and act during a turn.

# Player

### Action Space

The action space is discrete and varies based on the number of players. Actions are mapped to integers starting from 0:

|  Bit  |  Action             |  Target  |
|-------|---------------------|----------|
|  0    |  LOSE_AMBASSADOR    |  0       |
|  1    |  LOSE_ASSASSIN      |  0       |
|  2    |  LOSE_CAPTAIN       |  0       |
|  3    |  LOSE_CONTESSA      |  0       |
|  4    |  LOSE_DUKE          |  0       |
|  5    |  CHALLENGE_PASS     |  0       |
|  6    |  CHALLENGE_CALL     |  0       |
|  7    |  BLOCK_PASS         |  0       |
|  8    |  BLOCK_ASSASSINATE  |  0       |
|  9    |  BLOCK_FOREIGN_AID  |  0       |
|  10   |  BLOCK_STEAL_AMB    |  0       |
|  11   |  BLOCK_STEAL_CAP    |  0       |
|  12   |  EXCHANGE           |  0       |
|  13   |  FOREIGN_AID        |  0       |
|  14   |  INCOME             |  0       |
|  15   |  TAX                |  0       |
|  16   |  ASSASSINATE        |  1       |
|  17   |  ASSASSINATE        |  2       |
|  18   |  ASSASSINATE        |  3       |
|  19   |  ASSASSINATE        |  4       |
|  20   |  ASSASSINATE        |  5       |
|  21   |  COUP               |  1       |
|  22   |  COUP               |  2       |
|  23   |  COUP               |  3       |
|  24   |  COUP               |  4       |
|  25   |  COUP               |  5       |
|  26   |  STEAL              |  1       |
|  27   |  STEAL              |  2       |
|  28   |  STEAL              |  3       |
|  29   |  STEAL              |  4       |
|  30   |  STEAL              |  5       |

Here, target is the player position counting from the acting agent, so target=0 actions are self-targeted.

**Self-targeted actions**:
- INCOME: Gain 1 coin (cannot be blocked or challenged)
- FOREIGN_AID: Gain 2 coins (can be blocked by Duke)
- EXCHANGE: Claim Ambassador, draw 2 cards and return 2 cards of your choice (can be challenged)
- TAX: Claim Duke, gain 3 coins (can be challenged)

**Targeted actions**:
- COUP: Pay 7 coins to force a player to lose a card (cannot be blocked or challenged)
- ASSASSINATE: Claim Assassin, pay 3 coins to force player to lose a card (can be blocked by Contessa, can be challenged)
- STEAL: Claim Captain, steal 2 coins from a player (can be blocked by Captain or Ambassador, can be challenged)

**Response actions**:
- CHALLENGE_PASS: Do not challenge the action
- CHALLENGE_CALL: Challenge the action (available when an opponent claims a card)
- BLOCK_PASS: Don't block the action
- BLOCK_*: Various blocking actions
- LOSE_*: Choose which card to lose from your hand

### Legal Actions Mask

The legal actions available to the current agent are provided in the `action_mask` element of the info dictionary returned by `last()`.
The `action_mask` is a binary vector where each index represents whether the corresponding action is legal.

Taking an illegal action will raise a ValueError.

### Rewards

Coup uses a winner-takes-all reward structure:
- The last surviving player receives a reward of +1.0
- All other players receive 0.0

The game ends when only one player has remaining cards.

### Game Flow

A typical turn in Coup follows this pattern:

1. **Start State**: Active player chooses an action
2. **Challenge Window**: If action is challengeable, other players can challenge
3. **Block Window**: If action is blockable, target player (or all players) can block
4. **Block Challenge Window**: If blocked, other players can challenge the block
5. **Resolution**: Action resolves (or doesn't) based on challenges/blocks.
6. **End Turn**: Next player's turn begins

Players lose cards when:
- They lose a challenge
- They are successfully assassinated or coup'd

### Implementation

The game is modeled as a directed decision graph and implemented using a scrict state space. Here is a diagram of the game model:

![Game State Flow Diagram](https://github.com/user-attachments/assets/5e9bd45c-8002-4766-a0f0-cf2f0fb5d743)

### Version History

* v0: Initial release


