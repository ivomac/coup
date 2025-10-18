from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, auto

import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv


class InvalidState(Exception):
    pass


class CARD(Enum):
    AMBASSADOR = "Ambassador"
    ASSASSIN = "Assassin"
    CAPTAIN = "Captain"
    CONTESSA = "Contessa"
    DUKE = "Duke"

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


class ACT_TYPE(Enum):
    SELF = auto()
    TARGET = auto()
    BLOCK = auto()
    CHALLENGE = auto()
    LOSE = auto()

    def __repr__(self):
        return f"ACT_TYPE.{self.name}"

    def __str__(self):
        return repr(self)


@dataclass(slots=True)
class Player:
    id: int
    coins: int = 2
    cards: Counter[CARD] = field(default_factory=Counter)

    block_passed: bool = False
    challenge_passed: bool = False
    block_challenge_passed: bool = False

    @property
    def alive(self) -> bool:
        return self.cards.total() > 0

    def __repr__(self) -> str:
        cards = " ".join(str(card)[:3] for card in self.cards.elements())
        return f"Player {self.id} ({self.coins}$|{cards})"

    def __str__(self):
        return repr(self)


@dataclass(frozen=True, slots=True)
class Action:
    id: int
    type: ACT_TYPE
    name: str
    card: CARD | None = None
    block: tuple = field(default_factory=tuple)

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


ACTION_DICTS = [
    {"type": ACT_TYPE.CHALLENGE, "name": "CHALLENGE_PASS"},
    {"type": ACT_TYPE.CHALLENGE, "name": "CHALLENGE_CALL"},
    {"type": ACT_TYPE.BLOCK, "name": "BLOCK_PASS"},
    {"type": ACT_TYPE.BLOCK, "name": "BLOCK_ASSASSINATE", "card": CARD.CONTESSA},
    {"type": ACT_TYPE.BLOCK, "name": "BLOCK_FOREIGN_AID", "card": CARD.DUKE},
    {"type": ACT_TYPE.BLOCK, "name": "BLOCK_STEAL_AMB", "card": CARD.AMBASSADOR},
    {"type": ACT_TYPE.BLOCK, "name": "BLOCK_STEAL_CAP", "card": CARD.CAPTAIN},
    {"type": ACT_TYPE.SELF, "name": "EXCHANGE", "card": CARD.AMBASSADOR},
    {"type": ACT_TYPE.SELF, "name": "FOREIGN_AID", "block": ("BLOCK_FOREIGN_AID",)},
    {"type": ACT_TYPE.SELF, "name": "INCOME"},
    {"type": ACT_TYPE.SELF, "name": "TAX", "card": CARD.DUKE},
    {
        "type": ACT_TYPE.TARGET,
        "name": "ASSASSINATE",
        "card": CARD.ASSASSIN,
        "block": ("BLOCK_PASS", "BLOCK_ASSASSINATE"),
    },
    {"type": ACT_TYPE.TARGET, "name": "COUP"},
    {
        "type": ACT_TYPE.TARGET,
        "name": "STEAL",
        "card": CARD.CAPTAIN,
        "block": ("BLOCK_PASS", "BLOCK_STEAL_AMB", "BLOCK_STEAL_CAP"),
    },
]

for card in CARD:
    action = {"type": ACT_TYPE.LOSE, "name": f"LOSE_{card.name}", "card": card}
    ACTION_DICTS.append(action)


ACTION_BY_NAME: dict[str, Action] = {}
ACTION_BY_ID: list[Action] = []

for i, action_dict in enumerate(ACTION_DICTS):
    name = action_dict["name"]
    action = Action(id=i, **action_dict)

    ACTION_BY_NAME[name] = action
    ACTION_BY_ID.append(action)


@dataclass(frozen=True, slots=True)
class StartState:
    id: int
    player: Player


@dataclass(frozen=True, slots=True)
class ActionState(StartState):
    actor: Player
    action: Action
    target: Player


@dataclass(frozen=True, slots=True)
class ChallengeState(ActionState):
    pass


@dataclass(frozen=True, slots=True)
class ChallengeResolveState(ActionState):
    challenger: Player
    challenge_loser: Player


@dataclass(frozen=True, slots=True)
class BlockState(ActionState):
    challenger: Player | None = field(kw_only=True, default=None)
    challenge_loser: Player | None = field(kw_only=True, default=None)


@dataclass(frozen=True, slots=True)
class BlockChallengeState(BlockState):
    blocker: Player
    block: Action


@dataclass(frozen=True, slots=True)
class BlockChallengeResolveState(BlockChallengeState):
    block_challenger: Player
    block_challenge_loser: Player


@dataclass(frozen=True, slots=True)
class ActionResolveState(ActionState):
    challenger: Player | None = field(kw_only=True, default=None)
    challenge_loser: Player | None = field(kw_only=True, default=None)

    blocker: Player | None = field(kw_only=True, default=None)
    block: Action | None = field(kw_only=True, default=None)

    block_challenger: Player | None = field(kw_only=True, default=None)
    block_challenge_loser: Player | None = field(kw_only=True, default=None)

    counter: int


# TODO
"""
    def __repr__(self):
        s = f"Turn {self.id}: Player {self.actor.id}"

        if hasattr(self, "action"):
            s += f"\n  chose {self.action.name}"

        if self.target:
            s += f" on Player {self.target.id}"

        if self.challenger:
            s += f"\n  challenged by Player {self.challenger.id}"

            if self.challenge_loser == self.challenger:
                s += " > was not bluff"
            else:
                s += " > was bluff"

            if self.challenge_resolved:
                s += f" > {self.challenge_loser} losed a card"

        if self.blocker:
            s += f"\n  blocked by Player {self.blocker.id}"

            if self.block:
                s += f" with {self.block.name}"

        if self.block_challenger:
            s += f"\n  block challenged by Player {self.block_challenger.id}"

            if self.block_challenge_loser == self.block_challenger:
                s += " > was not bluff"
            else:
                s += " > was bluff"

            if self.block_challenge_resolved:
                s += f" > {self.block_challenge_loser} losed a card"

        return s

    def __str__(self):
        return repr(self)
    """


AnyState = (
    StartState
    | ActionState
    | ChallengeState
    | ChallengeResolveState
    | BlockState
    | BlockChallengeState
    | BlockChallengeResolveState
    | ActionResolveState
)


class raw_env(AECEnv):
    metadata = {
        "render_modes": ["ansi"],
        "name": "coup_v0",
    }

    def __init__(
        self,
        render_mode: str = "ansi",
        num_players: int = 6,
        num_players_alive: int = 6,
        deck: dict[str, int] | None = None,
        revealed: dict[str, int] | None = None,
        reveal_at_start: bool = False,
    ):
        super().__init__()

        self.game_id = 0

        self.render_mode: str = render_mode

        ### DECK ###

        self.base_deck: Counter[CARD]

        if not deck:
            self.base_deck = Counter(dict.fromkeys(CARD, 3))
        else:
            self.base_deck = Counter(
                {CARD(card_str): count for card_str, count in deck.items()}
            )

        self.deck: Counter[CARD]

        if not revealed:
            revealed = {}

        self.base_revealed: Counter[CARD] = Counter(
            {CARD(card_str): count for card_str, count in revealed.items()}
        )

        self.revealed: Counter[CARD]

        self.reveal_at_start: bool = reveal_at_start

        if num_players_alive > num_players:
            raise ValueError(
                f"{num_players_alive=} cannot be greater than {num_players=}"
            )

        max_players = self.base_deck.total() // 2 - 1

        if num_players_alive > max_players:
            raise ValueError(
                f"{num_players_alive=} cannot be greater than "
                f"{max_players=} with current deck"
            )

        ### PLAYERS ###

        self.num_players_alive: int = num_players_alive

        self.possible_agents: list[int] = list(range(num_players))
        self.agents: list[int]

        self.players: list[Player]

        ### GAME STATE ###

        self.turn: AnyState

        self.rewards: dict[int, float]

        self.infos: dict[int, dict]
        self.truncations: dict[int, bool]
        self.terminations: dict[int, bool]

        ### ACTION MAP ###

        self.idx_to_act: dict[int, tuple[Action, int]] = {}
        idx = 0

        for action in ACTION_BY_ID:
            if action.type != ACT_TYPE.TARGET:
                self.idx_to_act[idx] = (action, 0)
                idx += 1
            else:
                for target in range(1, len(self.possible_agents)):
                    self.idx_to_act[idx] = (action, target)
                    idx += 1

        self.act_to_idx = {v: k for k, v in self.idx_to_act.items()}

    @property
    def winner(self) -> int | None:
        alive = [player for player in self.players if player.alive]

        if len(alive) == 1:
            return alive[0].id

        if not alive:
            raise InvalidState("No players alive.")

        return None

    @property
    def agent_selection(self) -> int:
        return self.turn.player.id

    def enumerate_from(
        self, agent: int, alive: bool = False
    ) -> Iterator[tuple[int, Player]]:
        from_agent = self.players[agent:] + self.players[:agent]
        for i, player in enumerate(from_agent):
            if not alive or player.alive:
                yield i, player

    def enumerate_after(
        self, agent: int, alive: bool = False
    ) -> Iterator[tuple[int, Player]]:
        after_agent = self.players[agent + 1 :] + self.players[:agent]
        for i, player in enumerate(after_agent, start=1):
            if not alive or player.alive:
                yield i, player

    def draw_from_deck(self) -> CARD:
        cards = list(self.deck.elements())
        idx = np.random.randint(0, len(cards) - 1)
        card = cards[idx]
        self.deck[card] -= 1
        return card

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed or options:
            UserWarning("Arguments passed to reset method, but not used")

        self.game_id += 1

        self.deck = self.base_deck.copy()
        self.revealed = self.base_revealed.copy()

        self.agents = self.possible_agents[: self.num_players_alive]

        self.players = [Player(id=_id) for _id in self.possible_agents]

        for player in self.players:
            if player.id < len(self.agents):
                # Player is alive.
                for _ in range(2):
                    card = self.draw_from_deck()
                    player.cards[card] += 1
            elif self.reveal_at_start:
                # Player is dead. Reveal two cards?
                for _ in range(2):
                    card = self.draw_from_deck()
                    self.revealed[card] += 1

        self.turn = StartState(id=0, player=self.players[0])

        self.rewards = dict.fromkeys(self.possible_agents, 0.0)

    def observation_space(self, agent: int):
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} is not a valid agent.")

        max_card_count = max(self.base_deck.values())

        n_block_actions = sum(
            1 for action in ACTION_BY_ID if action.type == ACT_TYPE.BLOCK
        )
        n_block_actions -= 1

        n_first_actions = sum(
            1
            for action in ACTION_BY_ID
            if action.type in {ACT_TYPE.SELF, ACT_TYPE.TARGET}
        )

        return Dict(
            coins=MultiDiscrete(len(self.possible_agents) * [13]),
            cards=Dict(
                revealed=MultiDiscrete(len(CARD) * [max_card_count]),
                ours=MultiDiscrete(len(CARD) * [max_card_count]),
                total=MultiDiscrete(len(self.possible_agents) * [5]),
            ),
            action=Dict(
                idx=MultiBinary(n_first_actions),
                actor=MultiBinary(len(self.possible_agents)),
                target=MultiBinary(len(self.possible_agents)),
            ),
            challenge=Dict(
                passed=MultiBinary(len(self.possible_agents)),
                challenger=MultiBinary(len(self.possible_agents)),
                loser=MultiBinary(len(self.possible_agents)),
            ),
            block=Dict(
                idx=MultiBinary(n_block_actions),
                passed=MultiBinary(len(self.possible_agents)),
                blocker=MultiBinary(len(self.possible_agents)),
            ),
            block_challenge=Dict(
                passed=MultiBinary(len(self.possible_agents)),
                challenger=MultiBinary(len(self.possible_agents)),
                loser=MultiBinary(len(self.possible_agents)),
            ),
        )

    def observe(self, agent: int) -> dict:
        turn = self.turn

        block_actions = [
            action for action in ACTION_BY_ID if action.type == ACT_TYPE.BLOCK
        ]

        first_actions = [
            action
            for action in ACTION_BY_ID
            if action.type in {ACT_TYPE.SELF, ACT_TYPE.TARGET}
        ]

        def attr(string):
            return getattr(turn, string, None)

        return {
            "coins": [player.coins for _, player in self.enumerate_from(agent)],
            "cards": {
                "revealed": [self.revealed[card] for card in CARD],
                "ours": [self.players[agent].cards[card] for card in CARD],
                "total": [
                    player.cards.total() for _, player in self.enumerate_from(agent)
                ],
            },
            "action": {
                "idx": [attr("action") == action for action in first_actions],
                "actor": [
                    attr("actor") == player for _, player in self.enumerate_from(agent)
                ],
                "target": [
                    attr("target") == player for _, player in self.enumerate_from(agent)
                ],
            },
            "challenge": {
                "passed": [
                    player.challenge_passed for _, player in self.enumerate_from(agent)
                ],
                "challenger": [
                    attr("challenger") == player
                    for _, player in self.enumerate_from(agent)
                ],
                "loser": [
                    attr("challenge_loser") == player
                    for _, player in self.enumerate_from(agent)
                ],
            },
            "block": {
                "idx": [attr("block") == action for action in block_actions],
                "passed": [
                    player.block_passed for _, player in self.enumerate_from(agent)
                ],
                "blocker": [
                    attr("blocker") == player
                    for _, player in self.enumerate_from(agent)
                ],
            },
            "block_challenge": {
                "passed": [
                    player.block_challenge_passed
                    for _, player in self.enumerate_from(agent)
                ],
                "challenger": [
                    attr("block_challenger") == player
                    for _, player in self.enumerate_from(agent)
                ],
                "loser": [
                    attr("block_challenge_loser") == player
                    for _, player in self.enumerate_from(agent)
                ],
            },
        }

    def action_space(self, agent: int):
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} is not a valid agent.")
        return Discrete(len(self.idx_to_act))

    def get_action_mask(self) -> np.ndarray:
        action_mask: np.ndarray = np.zeros(len(self.idx_to_act), dtype=np.int8)

        def mask(action: Action, target: int = 0):
            action_mask[self.act_to_idx[(action, target)]] = 1

        def mask_type(tp: ACT_TYPE):
            for action in ACTION_BY_ID:
                if action.type == tp:
                    mask(action)

        def mask_lose_type(loser):
            for action in ACTION_BY_ID:
                if action.type == ACT_TYPE.LOSE and loser.cards[action.card]:
                    mask(action)

        if isinstance(self.turn, StartState):
            player = self.turn.player

            if player.coins >= 7:
                for i, _ in self.enumerate_after(player.id, alive=True):
                    mask(ACTION_BY_NAME["COUP"], i)

                if player.coins >= 10:
                    # must coup
                    return action_mask

            for i, other in self.enumerate_after(player.id, alive=True):
                if player.coins >= 7:
                    mask(ACTION_BY_NAME["ASSASSINATE"], i)
                if other.coins >= 1:
                    mask(ACTION_BY_NAME["STEAL"], i)

            mask_type(ACT_TYPE.SELF)

        elif isinstance(self.turn, ChallengeState | BlockChallengeState):
            mask_type(ACT_TYPE.CHALLENGE)

        elif isinstance(self.turn, ChallengeResolveState):
            mask_lose_type(self.turn.challenge_loser)

        elif isinstance(self.turn, BlockState):
            for block_name in list(self.turn.action.block):
                mask(ACTION_BY_NAME[block_name])

        elif isinstance(self.turn, BlockChallengeResolveState):
            mask_lose_type(self.turn.block_challenge_loser)

        elif isinstance(self.turn, ActionResolveState):
            mask_lose_type(self.turn.target)

        return action_mask

    def last(self, observe: bool = True):
        player = self.turn.player
        action_mask = self.get_action_mask()

        observation = self.observe(player.id) if observe else {}

        reward = self.rewards[player.id]

        termination = self.winner is not None

        if not action_mask.any() and not termination:
            raise ValueError("No available actions")

        return (
            observation,
            reward,
            termination,
            False,
            {"action_mask": action_mask},
        )

    def step_choose_action(self, action: Action, target: Player):
        assert isinstance(self.turn, StartState)

        self.turn = ActionState(
            **asdict(self.turn),
            action=action,
            actor=self.turn.player,
            target=target,
        )

        if self.turn.action == ACTION_BY_NAME["INCOME"]:
            self.turn.actor.coins += 1
            self.turn = ActionResolveState(**asdict(self.turn), counter=0)

        elif self.turn.action == ACTION_BY_NAME["FOREIGN_AID"]:
            self.turn = BlockState(
                **asdict(self.turn), player=self.next(self.turn.actor)
            )

        elif self.turn.action == ACTION_BY_NAME["COUP"]:
            self.turn.actor.coins -= 7

        elif self.turn.action == ACTION_BY_NAME["ASSASSINATE"]:
            self.turn.actor.coins -= 3

        if self.turn.action.card:
            self.turn = ChallengeState(
                **asdict(self.turn), player=self.next(self.turn.actor)
            )

        elif self.turn.action.block:
            self.turn = BlockState(**asdict(self.turn), player=self.turn.target)

    def step_challenge_action(self, action):
        player = self.turn.player

        assert isinstance(self.turn, ChallengeState)

        if action == ACTION_BY_NAME["CHALLENGE_PASS"]:
            player.challenge_passed = True
            self.turn = replace(self.turn, player=self.next(player))

            if self.turn.player == self.turn.actor:
                if self.turn.action == ACTION_BY_NAME["EXCHANGE"]:
                    for _ in range(2):
                        card = self.draw_from_deck()
                        self.turn.actor.cards[card] += 1

                elif self.turn.action == ACTION_BY_NAME["TAX"]:
                    self.turn.actor.coins += 3
                    self.turn = ActionResolveState(**asdict(self.turn), counter=0)

                if self.turn.action.block:
                    self.turn = BlockState(**asdict(self.turn), player=self.turn.target)

        elif action == ACTION_BY_NAME["CHALLENGE_CALL"]:
            action_card = self.turn.action.card
            if action_card and self.turn.actor.cards[action_card]:
                loser = player

                self.turn.actor.cards[action_card] -= 1
                self.deck[action_card] += 1
                new_card = self.draw_from_deck()
                self.turn.actor.cards[new_card] += 1

            else:
                loser = self.turn.actor

            self.turn = ChallengeResolveState(
                **asdict(self.turn),
                challenger=player,
                challenge_loser=loser,
                player=loser,
            )

    def step_challenge_resolve_action(self, action):
        assert isinstance(self.turn, ActionResolveState)
        assert action.type == ACT_TYPE.LOSE

        card = action.card
        player = self.turn.player

        player.cards[card] -= 1
        self.revealed[card] += 1

        if self.turn.challenge_loser == self.turn.actor:
            self.turn = ActionResolveState(**asdict(self.turn), counter=0)

        elif (
            self.turn.target
            and not self.turn.target.alive
            and self.turn.action == ACTION_BY_NAME["STEAL"]
        ):
            amount = min(2, self.turn.target.coins)
            self.turn.actor.coins += amount
            self.turn.target.coins -= amount
            self.turn = ActionResolveState(**asdict(self.turn), counter=0)

        elif self.turn.action == ACTION_BY_NAME["EXCHANGE"]:
            for _ in range(2):
                card = self.draw_from_deck()
                self.turn.actor.cards[card] += 1

        elif self.turn.action.block:
            self.turn = BlockState(**asdict(self.turn), player=self.turn.target)

        raise InvalidState("Action not handled for this state.")

    def step_block_action(self, action: Action):
        player = self.turn.player

        if action == ACTION_BY_NAME["BLOCK_PASS"]:
            player.block_passed = True

            if turn.action == ACTION_BY_NAME["FOREIGN_AID"]:
                turn.blocker_candidate = self.next(player)

                if turn.blocker_candidate == turn.actor:
                    turn.blocker_candidate = None

                    turn.actor.coins += 2
                    turn.action_resolved = True

            else:
                turn.blocker_candidate = None

            if turn.action == ACTION_BY_NAME["STEAL"]:
                amount = min(2, turn.target.coins)
                turn.actor.coins += amount
                turn.target.coins -= amount
                turn.action_resolved = True

        else:
            turn.block = action
            turn.blocker = player
            turn.blocker_candidate = None
            turn.block_challenger_candidate = self.next(player)

    def step_challenge_block(self, action: Action):
        player = self.turn.player

        if action == ACTION_BY_NAME["CHALLENGE_PASS"]:
            player.block_challenge_passed = True
            turn.block_challenger_candidate = self.next(player)

            if turn.block_challenger_candidate == turn.actor:
                turn.action_resolved = True
                turn.block_challenge_resolved = True
                turn.block_challenger_candidate = None

        elif action == ACTION_BY_NAME["CHALLENGE_CALL"]:
            turn.block_challenger = turn.block_challenger_candidate
            turn.block_challenger_candidate = None

            if turn.block_challenge_loser != turn.blocker:
                turn.blocker.cards[turn.block.card] -= 1
                self.deck[turn.block.card] += 1
                new_card = self.draw_from_deck()
                turn.blocker.cards[new_card] += 1

        elif action.type == ACT_TYPE.LOSE:
            card = action.card

            player.cards[card] -= 1
            self.revealed[card] += 1

            turn.block_challenge_resolved = True
            if turn.block_challenge_loser == turn.block_challenger:
                turn.action_resolved = True

            elif turn.action == ACTION_BY_NAME["STEAL"]:
                amount = min(2, turn.target.coins)
                turn.actor.coins += amount
                turn.target.coins -= amount
                turn.action_resolved = True

        else:
            raise InvalidState("Action not handled for this state.")

    def step_lose_card(self, action: Action):
        player = self.turn.player

        card = action.card

        if turn.action == ACTION_BY_NAME["EXCHANGE"]:
            player.cards[card] -= 1
            self.deck[card] += 1

            if turn.action_resolved is None:
                turn.action_resolved = False
            else:
                turn.action_resolved = True
        else:
            player.cards[card] -= 1
            self.revealed[card] += 1

            turn.action_resolved = True

    def step(self, action: int):
        turn = self.turn
        player = turn.player

        action_mask = self.get_action_mask()

        if not action_mask[action]:
            raise ValueError(f"Action {action} is not an allowed action.")

        action_obj, tgt = self.idx_to_act[action]
        target = self.players[(player.id + tgt) % len(self.players)]

        if isinstance(turn, StartState):
            self.step_choose_action(action_obj, target)
        elif turn.action.card and not turn.challenge_resolved:
            self.step_challenge_action(action_obj)
        elif turn.action.block and turn.blocker_candidate:
            self.step_block_action(action_obj)
        elif turn.block and not turn.block_challenge_resolved:
            self.step_challenge_block(action_obj)
        elif action_obj.type == ACT_TYPE.LOSE:
            self.step_lose_card(action_obj)
        else:
            raise InvalidState("Action not handled for this state.")

        if turn.action_resolved:
            for player in self.players:
                player.challenge_passed = False
                player.block_passed = False
                player.block_challenge_passed = False

            next_player = self.next(self.turn.actor)
            turn_number = self.turn.id
            self.turn = StartState(id=turn_number, player=next_player)

    def render(self):
        print("Turn:", self.turn.id)
        print("Actor:", self.turn.actor)
        for player in self.players:
            print(player)

    def next(self, player: Player):
        for _, other in self.enumerate_after(player.id, alive=True):
            return other
        return player
