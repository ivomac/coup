import random
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import IntEnum, auto

import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from pettingzoo import AECEnv


class CARD(IntEnum):
    AMBASSADOR = auto()
    ASSASSIN = auto()
    CAPTAIN = auto()
    CONTESSA = auto()
    DUKE = auto()


# In Priority Order
class PHASE(IntEnum):
    ACTION_CHALLENGE_RESOLVE = auto()
    BLOCK_CHALLENGE_RESOLVE = auto()
    ACTION_CHALLENGE = auto()
    BLOCK_CHALLENGE = auto()
    BLOCK = auto()
    ACTION_RESOLVE = auto()
    ACTION = auto()


class ACTION(IntEnum):
    # Actions that can only target self or original actor
    # TARGETED ACTION
    ASSASSINATE = auto()
    COUP = auto()
    STEAL = auto()

    # ACTION
    EXCHANGE = auto()
    FOREIGN_AID = auto()
    INCOME = auto()
    TAX = auto()

    # CHALLENGE
    CHALLENGE_PASS = auto()
    CHALLENGE_CALL = auto()

    # BLOCK
    BLOCK_PASS = auto()
    BLOCK_ASSASSINATE = auto()
    BLOCK_FOREIGN_AID = auto()
    BLOCK_STEAL_AMBASSADOR = auto()
    BLOCK_STEAL_CAPTAIN = auto()

    # CARD RESOLVE/PUTBACK
    LOSE_AMBASSADOR = auto()
    LOSE_ASSASSIN = auto()
    LOSE_CAPTAIN = auto()
    LOSE_CONTESSA = auto()
    LOSE_DUKE = auto()


LOSE_INFO: dict[CARD, ACTION] = {
    CARD.AMBASSADOR: ACTION.LOSE_AMBASSADOR,
    CARD.ASSASSIN: ACTION.LOSE_ASSASSIN,
    CARD.CAPTAIN: ACTION.LOSE_CAPTAIN,
    CARD.CONTESSA: ACTION.LOSE_CONTESSA,
    CARD.DUKE: ACTION.LOSE_DUKE,
}

ACTION_INFO: dict[ACTION, dict] = {
    ACTION.ASSASSINATE: {
        "card": CARD.ASSASSIN,
        "block": [ACTION.BLOCK_PASS, ACTION.BLOCK_ASSASSINATE],
        "targeted": True,
    },
    ACTION.COUP: {
        "targeted": True,
    },
    ACTION.STEAL: {
        "card": CARD.CAPTAIN,
        "block": [
            ACTION.BLOCK_PASS,
            ACTION.BLOCK_STEAL_AMBASSADOR,
            ACTION.BLOCK_STEAL_CAPTAIN,
        ],
        "targeted": True,
    },
    ACTION.EXCHANGE: {"card": CARD.AMBASSADOR},
    ACTION.FOREIGN_AID: {
        "block": [ACTION.BLOCK_PASS, ACTION.BLOCK_FOREIGN_AID],
    },
    ACTION.INCOME: {},
    ACTION.TAX: {"card": CARD.DUKE},
    ACTION.CHALLENGE_PASS: {},
    ACTION.CHALLENGE_CALL: {},
    ACTION.BLOCK_PASS: {},
    ACTION.BLOCK_ASSASSINATE: {"card": CARD.CONTESSA},
    ACTION.BLOCK_FOREIGN_AID: {"card": CARD.DUKE},
    ACTION.BLOCK_STEAL_AMBASSADOR: {"card": CARD.AMBASSADOR},
    ACTION.BLOCK_STEAL_CAPTAIN: {"card": CARD.CAPTAIN},
    ACTION.LOSE_AMBASSADOR: {"card": CARD.AMBASSADOR},
    ACTION.LOSE_ASSASSIN: {"card": CARD.ASSASSIN},
    ACTION.LOSE_CAPTAIN: {"card": CARD.CAPTAIN},
    ACTION.LOSE_CONTESSA: {"card": CARD.CONTESSA},
    ACTION.LOSE_DUKE: {"card": CARD.DUKE},
}


@dataclass(slots=True)
class Player:
    id: int
    deck: Counter[CARD] = field(repr=False)
    coins: int = 2
    cards: Counter[CARD] = field(default_factory=Counter)
    revealed: Counter[CARD] = field(default_factory=Counter)

    @property
    def alive(self) -> bool:
        return self.cards.total() > 0

    @property
    def name(self):
        return f"PLAYER {self.id}"

    def draw(self) -> CARD:
        cards = list(self.deck.elements())
        card = random.choice(cards)
        self.deck[card] -= 1
        self.cards[card] += 1
        return card

    def putback(self, card: CARD):
        self.cards[card] -= 1
        self.deck[card] += 1

    def reveal(self, card: CARD):
        self.cards[card] -= 1
        self.revealed[card] += 1


class CoupEnv(AECEnv):
    metadata = {
        "name": "coup_v0",
    }

    def __init__(
        self,
        num_agents: int = 6,
        max_turns: int = 60,
        turn_history_length: int = 12,
        reveal_at_start: bool = False,
        deck: dict[CARD, int] | None = None,
    ):
        self.game_id = 0

        if not deck:
            deck = dict.fromkeys(CARD, 3)
        self.base_deck = Counter(deck)
        self.max_num_card_type = max(self.base_deck.values())

        max_num_agents: int = self.base_deck.total() // 2 - 1
        if num_agents > max_num_agents:
            raise ValueError(
                f"{num_agents=} cannot be greater than"
                + f" {max_num_agents=} with current deck"
            )

        self.possible_agents: list[int] = list(range(max_num_agents))

        self.options = {"num_agents": num_agents}

        self.reveal_at_start: bool = reveal_at_start

        self.max_turns: int = max_turns
        self.turn_history_length: int = turn_history_length

        self.turn: int

        self.queue: dict[PHASE, deque[int]]

        self.deck: Counter[CARD]
        self.players: list[Player]

        self.agents: list[int]

        self.observations: dict[int, list[dict]]

        ### ACTION SPACE ###

        self.idx_to_action = {}
        idx = 0

        for act in ACTION:
            if "targeted" in ACTION_INFO[act]:
                for target in range(1, self.max_num_agents):
                    self.idx_to_action[idx] = (act, target)
                    idx += 1
            else:
                self.idx_to_action[idx] = (act, None)
                idx += 1

        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}

        self.action_space_size = len(self.idx_to_action)
        self.action_spaces = {agent: Discrete(idx) for agent in self.agents}

        self.num_cards = len(CARD)
        self.num_phases = len(PHASE)

        ### OBSERVATION SPACE ###

        self.num_observed_actions = len(ACTION)

        self.observation_turn_space = Dict(
            {
                "turn": Discrete(self.max_turns),
                "phases": Dict(
                    {
                        "done": MultiBinary(self.num_phases),
                        "current": MultiBinary(self.num_phases),
                    }
                ),
                "coins": MultiDiscrete(self.max_num_agents * [13], dtype=np.uint8),
                "cards": Dict(
                    {
                        "revealed": MultiDiscrete(
                            self.num_cards * [self.max_num_card_type], dtype=np.uint8
                        ),
                        "ours": MultiDiscrete(
                            self.num_cards * [self.max_num_card_type], dtype=np.uint8
                        ),
                        "total": MultiDiscrete(self.num_agents * [5], dtype=np.uint8),
                    }
                ),
                "action": Dict(
                    {
                        "idx": Discrete(self.num_observed_actions),
                        "card": MultiBinary(self.num_cards),
                        "passed": MultiBinary(self.max_num_agents),
                        "actor": MultiBinary(self.max_num_agents),
                        "target": MultiBinary(self.max_num_agents),
                        "successful": Discrete(2),
                    }
                ),
                "challenge": Dict(
                    {
                        "passed": MultiBinary(self.max_num_agents),
                        "challenger": MultiBinary(self.max_num_agents),
                        "loser": MultiBinary(self.max_num_agents),
                    }
                ),
            }
        )

        self.observation_agent_space = Tuple(
            self.observation_turn_space for _ in range(self.turn_history_length)
        )

        self.observation_space_size = self.observation_agent_space.shape
        self.observation_spaces = dict.fromkeys(
            self.possible_agents, self.observation_agent_space
        )

    @property
    def phase(self) -> PHASE:
        for phase in PHASE:
            if self.queue[phase]:
                return phase
        raise Exception("No game phase is active.")

    @property
    def agent_selection(self) -> int:
        for phase in PHASE:
            if self.queue[phase]:
                return self.queue[phase][0]
        raise Exception("No agent is selected.")

    def enumerate_from(self, agent: int) -> Iterator[tuple[int, Player]]:
        from_agent = self.players[agent:] + self.players[:agent]
        yield from enumerate(from_agent)

    def enumerate_after(self, agent: int) -> Iterator[tuple[int, Player]]:
        after_agent = self.players[agent + 1 :] + self.players[:agent]
        yield from enumerate(after_agent, start=1)

    def new_observation(self, agent: int) -> dict:
        revealed = Counter()
        for player in self.players:
            revealed += player.revealed

        return {
            "turn": self.turn,
            "phases": {
                "done": [False for _ in PHASE],
                "current": [False for _ in PHASE],
            },
            "coins": [player.coins for _, player in self.enumerate_from(agent)],
            "cards": {
                "revealed": [revealed[card] for card in CARD],
                "ours": [self.players[agent].cards[card] for card in CARD],
                "total": [
                    player.cards.total() for _, player in self.enumerate_from(agent)
                ],
            },
            "action": {
                "idx": [False for _ in range(self.num_observed_actions)],
                "card": [False for _ in range(self.num_cards)],
                "passed": [False for _ in range(self.max_num_agents)],
                "actor": [False for _ in range(self.max_num_agents)],
                "target": [False for _ in range(self.max_num_agents)],
                "successful": False,
            },
            "challenge": {
                "passed": [False for _ in range(self.max_num_agents)],
                "challenger": [False for _ in range(self.max_num_agents)],
                "loser": [False for _ in range(self.max_num_agents)],
            },
        }

    def observe(self, agent: int):
        return tuple(self.observations[agent][-self.turn_history_length :])

    def winner(self) -> int | None:
        alive = [player for player in self.players if player.alive]
        return alive[0].id if len(alive) == 1 else None

    def reset(self, seed: int | None = None, options: dict | None = None):
        if options:
            UserWarning(f"options passed to reset method, but not used: {options}")

        self.game_id += 1
        random.seed(seed)

        num_agents = self.options["num_agents"]
        self.agents = self.possible_agents[:num_agents]

        self.actor_queue = deque(self.agents)
        self.blocker_queue = deque()
        self.challenger_queue = deque()

        self.deck = self.base_deck.copy()
        self.players = [Player(id=id, deck=self.deck) for id in self.possible_agents]

        for player in self.players:
            if player.id < len(self.agents):
                player.draw()
                player.draw()
            elif self.reveal_at_start:
                player.draw()
                player.draw()
                for card in player.cards.elements():
                    player.reveal(card)

        self.turn = 0

        self.queue = {phase: deque() for phase in PHASE}

        self.observations = {
            agent: [
                self.new_observation(agent) for _ in range(self.turn_history_length)
            ]
            for agent in self.possible_agents
        }

        for agent in self.agents:
            self.observations[agent][-1]["phase"][PHASE.ACTION.value] = True

        self.rewards = dict.fromkeys(self.possible_agents, 0)

        self.terminations = {
            agent: agent not in self.agents for agent in self.possible_agents
        }

        self.truncations = dict.fromkeys(self.possible_agents, False)

        self.infos = {
            agent: {"action_mask": self.action_mask(agent)}
            for agent in self.possible_agents
        }

    def action_mask(self, agent: int) -> list[bool]:
        mask: list[bool] = [False] * self.action_space_size

        if agent != self.agent_selection:
            return mask

        actor = self.players[agent]
        obs = self.observations[agent]
        phase = PHASE(obs[-1]["phase"]["current"].index(True))

        match phase:
            case PHASE.ACTION:
                if actor.coins >= 7:
                    for i, other in self.enumerate_after(agent):
                        if other.alive:
                            idx = self.action_to_idx[(ACTION.COUP, i)]
                            mask[idx] = True
                    if actor.coins >= 10:
                        return mask

                if actor.coins >= 3:
                    for i, other in self.enumerate_after(agent):
                        if other.alive:
                            idx = self.action_to_idx[(ACTION.ASSASSINATE, i)]
                            mask[idx] = True

                for i, other in self.enumerate_after(agent):
                    if other.coins > 0:
                        idx = self.action_to_idx[(ACTION.STEAL, i)]
                        mask[idx] = True

                for act in [
                    ACTION.EXCHANGE,
                    ACTION.FOREIGN_AID,
                    ACTION.INCOME,
                    ACTION.TAX,
                ]:
                    idx = self.action_to_idx[(act, None)]
                    mask[idx] = True

            case PHASE.ACTION_CHALLENGE | PHASE.BLOCK_CHALLENGE:
                for act in [ACTION.CHALLENGE_PASS, ACTION.CHALLENGE_CALL]:
                    idx = self.action_to_idx[(act, None)]
                    mask[idx] = True
            case PHASE.BLOCK:
                action_turn = obs[-2]
                idx = action_turn["action"]["idx"].index(True)
                action = self.idx_to_action[idx]
                for reaction in ACTION_INFO[action]["block"]:
                    react_idx = self.action_to_idx[(reaction, None)]
                    mask[react_idx] = True
            case PHASE.ACTION_CHALLENGE_RESOLVE | PHASE.BLOCK_CHALLENGE_RESOLVE:
                for card in set(actor.cards.elements()):
                    action = LOSE_INFO[card]
                    lose_idx = self.action_to_idx[(action, card)]
                    mask[lose_idx] = True

        return mask

    def step(self, action: int):
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward
            self.rewards[agent] = 0

        agent = self.agent_selection
        actor = self.players[agent]
        act, target = self.idx_to_action[action]

        act_info = ACTION_INFO[act]

        agent_obs = self.observations[agent]

        obs = agent_obs[-1]
        phase = PHASE(obs["phase"]["current"].index(True))

        match phase:
            case PHASE.ACTION:
                obs["action"]["idx"][act] = True
                if target is None:
                    obs["action"]["target"][0] = True
                else:
                    obs["action"]["target"][target] = True

                if "card" in act_info:
                    card = act_info["card"]
                    obs["action"]["card"][card.value] = True
                    next_phase = PHASE.ACTION_CHALLENGE
                elif "block" in act_info:
                    next_phase = PHASE.BLOCK
                else:
                    next_phase = PHASE.ACTION

                if act == ACTION.ASSASSINATE:
                    actor.coins -= 3
                elif act == ACTION.COUP:
                    actor.coins -= 7
                    next_phase = PHASE.ACTION_RESOLVE

                obs["phases"]["done"][phase.value] = True
                obs["phases"]["current"][phase.value] = False
                obs["phases"]["current"][next_phase.value] = True

        # return obs, 0, self.winner() is not None, self.turn > self.max_turns, info

    def render(self):
        pass

    def observation_space(self, agent: int):
        return self.observation_spaces[agent]

    def action_space(self, agent: int):
        return self.action_spaces[agent]
