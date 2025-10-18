import logging
from collections import Counter

import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiBinary, MultiDiscrete
from pettingzoo import AECEnv

import coup_pettingzoo.env.state as State
from coup_pettingzoo.env.action import ACT, ACTION, Action
from coup_pettingzoo.env.card import CARD, draw
from coup_pettingzoo.env.player import Player, enum, nxt

logger = logging.getLogger(__name__)


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
        dead_draw: bool = False,
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

        self.dead_draw: bool = dead_draw

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

        self.n_turn: int
        self.turn: State.Any

        self.rewards: dict[int, float]

        self.infos: dict[int, dict]
        self.truncations: dict[int, bool]
        self.terminations: dict[int, bool]

        ### ACTION MAP ###

        self.idx_to_act: dict[int, tuple[Action, int]] = {}
        idx = 0

        for action in ACTION.iter():
            if action.type != ACT.TARGET:
                self.idx_to_act[idx] = (action, 0)
                idx += 1
            else:
                for target in range(1, len(self.possible_agents)):
                    self.idx_to_act[idx] = (action, target)
                    idx += 1

        self.act_to_idx = {v: k for k, v in self.idx_to_act.items()}

    @property
    def winner(self) -> Player | None:
        alive = [player for player in self.players if player.alive]

        if len(alive) == 1:
            return alive[0]

        if not alive:
            raise State.InvalidState("No players alive.")

        return None

    @property
    def agent_selection(self) -> int:
        return self.turn.player.id

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed or options:
            UserWarning("Arguments passed to reset method, but not used")
            logger.warning(
                "Arguments passed to reset method, but not used: seed=%s, options=%s",
                seed,
                options,
            )

        self.game_id += 1
        logger.debug(
            "Environment reset: game_id=%d, num_players_alive=%d, num_players=%d",
            self.game_id,
            self.num_players_alive,
            len(self.possible_agents),
        )

        self.deck = self.base_deck.copy()
        logger.debug("Initial deck: %s", {str(k): v for k, v in self.deck.items()})

        self.agents = self.possible_agents[: self.num_players_alive]
        logger.debug("Active agents: %s", self.agents)

        self.players = [
            Player(id=agent, _deck=self.deck) for agent in self.possible_agents
        ]

        for player in self.players:
            if player.id < len(self.agents):
                # Player is alive.
                logger.debug("Initializing alive player %d", player.id)
                for _ in range(2):
                    player.draw()
            elif self.dead_draw:
                # Player is dead. Remove two cards?
                logger.debug("Dead draw for player %d", player.id)
                for _ in range(2):
                    draw(self.deck)

        self.n_turn = 1
        self.turn = State.Start(player=self.players[0])
        logger.debug("Initial turn: player %d", self.turn.player.id)

        self.rewards = dict.fromkeys(self.possible_agents, 0.0)
        logger.debug("Initial rewards: %s", self.rewards)

    def observation_space(self, agent: int):
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} is not a valid agent.")

        max_card_count = max(self.base_deck.values())

        n_block_actions = sum(1 for action in ACTION.iter() if action.type == ACT.BLOCK)
        n_block_actions -= 1

        n_first_actions = sum(
            1 for action in ACTION.iter() if action.type in {ACT.SELF, ACT.TARGET}
        )

        return Dict(
            coins=MultiDiscrete(len(self.possible_agents) * [13]),
            cards=Dict(
                unseen=MultiDiscrete(len(CARD) * [max_card_count]),
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

        block_actions = [action for action in ACTION.iter() if action.type == ACT.BLOCK]

        first_actions = [
            action for action in ACTION.iter() if action.type in {ACT.SELF, ACT.TARGET}
        ]

        unseen = self.deck.copy()
        for _, player in enum(self.players, agent, alive=True, skip_self=True):
            for card in player.cards.elements():
                unseen[card] += 1

        def attr(string):
            return getattr(turn, string, None)

        return {
            "coins": [player.coins for _, player in enum(self.players, agent)],
            "cards": {
                "unseen": [unseen[card] for card in CARD],
                "ours": [self.players[agent].cards[card] for card in CARD],
                "total": [
                    player.cards.total() for _, player in enum(self.players, agent)
                ],
            },
            "action": {
                "idx": [attr("action") == action for action in first_actions],
                "actor": [
                    attr("actor") == player for _, player in enum(self.players, agent)
                ],
                "target": [
                    attr("target") == player for _, player in enum(self.players, agent)
                ],
            },
            "challenge": {
                "passed": [
                    player.challenge_passed for _, player in enum(self.players, agent)
                ],
                "challenger": [
                    attr("challenger") == player
                    for _, player in enum(self.players, agent)
                ],
                "loser": [
                    attr("challenge_loser") == player
                    for _, player in enum(self.players, agent)
                ],
            },
            "block": {
                "idx": [attr("block") == action for action in block_actions],
                "passed": [
                    player.block_passed for _, player in enum(self.players, agent)
                ],
                "blocker": [
                    attr("blocker") == player for _, player in enum(self.players, agent)
                ],
            },
            "block_challenge": {
                "passed": [
                    player.block_challenge_passed
                    for _, player in enum(self.players, agent)
                ],
                "challenger": [
                    attr("block_challenger") == player
                    for _, player in enum(self.players, agent)
                ],
                "loser": [
                    attr("block_challenge_loser") == player
                    for _, player in enum(self.players, agent)
                ],
            },
        }

    def action_space(self, agent: int):
        if agent not in self.possible_agents:
            raise ValueError(f"Agent {agent} is not a valid agent.")
        return Discrete(len(self.idx_to_act))

    def get_action_mask(self) -> np.ndarray:
        action_mask: np.ndarray = np.zeros(len(self.idx_to_act), dtype=np.int8)

        actions = self.turn.action_mask(self.players)

        for act, tgt in actions:
            idx = self.act_to_idx[(act, tgt)]
            action_mask[idx] = 1

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

    def step(self, action: int):
        turn = self.turn
        player = turn.player

        action_mask = self.get_action_mask()

        if not action_mask[action]:
            logger.error(
                "Invalid action %d for player %d in state %s",
                action,
                player.id,
                type(turn).__name__,
            )
            raise ValueError(f"Action {action} is not an allowed action.")

        action_obj, tgt = self.idx_to_act[action]
        target = self.players[(player.id + tgt) % len(self.players)]

        logger.debug(
            "Step: turn_type=%s, player=%d, action=%s, target=%d",
            type(turn).__name__,
            player.id,
            action_obj.name,
            target.id,
        )

        self.turn = self.turn.step(action_obj, target, self.players)
        logger.debug("New turn state: %s", type(self.turn).__name__)

        if type(self.turn) is State.EndTurn:
            logger.debug("End of turn reached, resetting player states")
            for player in self.players:
                player.challenge_passed = False
                player.block_passed = False
                player.block_challenge_passed = False

            if self.winner:
                logger.debug(
                    "Player %d won with %d cards",
                    self.winner.id,
                    self.winner.cards.total(),
                )
                next_player = self.winner
            else:
                next_player = nxt(self.players, self.turn.act.actor.id)

            logger.debug(
                "Turn %d completed, moving to player %d", self.n_turn, next_player.id
            )
            self.n_turn += 1
            self.turn = State.Start(player=next_player)

    def render(self):
        if self.winner:
            print(f"\nGAME OVER - Winner is Player {self.winner.id}\n")
            return

        print(
            f"\nTurn {self.n_turn}"
            + f" - {type(self.turn).__name__}"
            + f" - Player {self.turn.player.id}"
        )

        actions = self.turn.action_mask(self.players)

        available_actions = []
        for act, tgt in actions:
            if tgt == 0:
                available_actions.append(f"{act.name}")
            else:
                available_actions.append(f"{act.name}_{tgt}")

        print("Available actions:", " ".join(available_actions))

        max_len = {
            "id": len("id"),
            "cards": len("cards"),
            "coins": len("coins"),
        }

        players = [
            {
                "id": str(player.id),
                "cards": " ".join(str(card)[:3] for card in player.cards.elements()),
                "coins": f"{player.coins}$",
            }
            for player in self.players
        ]

        for player in players:
            for key, val in player.items():
                max_len[key] = max(max_len[key], len(val))

        row = ""
        for key, val in max_len.items():
            row += f"{key:<{val}} | "
        print(row)

        for player in players:
            row = ""
            for key, val in max_len.items():
                row += f"{player[key]:<{val}} | "
            print(row)
        print()
