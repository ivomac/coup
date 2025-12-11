"""PettingZoo environment implementation of the card game Coup."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv

from pettingzoo_coup.env import action, state
from pettingzoo_coup.env.action import ActionSpace
from pettingzoo_coup.env.card import CARD, draw
from pettingzoo_coup.env.player import AgentID, Player
from pettingzoo_coup.env.tabulate import to_markdown

logger = logging.getLogger(__name__)


class raw_env(AECEnv):
    """PettingZoo Coup game environment for RL training."""

    metadata = {
        "render_modes": ["ansi"],
        "name": "coup_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        deck: dict[str, int] | None = None,
        dead_draw: bool = False,
        num_players: int = 6,
        num_players_alive: int = 6,
        render_mode: str | None = "ansi",
    ):
        super().__init__()

        self.game_id = 0

        self.render_mode: str | None = render_mode

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

        if num_players < 2:
            raise ValueError(f"{num_players=} cannot be smaller than 2")

        if num_players_alive < 2:
            raise ValueError(f"{num_players_alive=} must be greater than 1")

        if num_players_alive > num_players:
            raise ValueError(
                f"{num_players_alive=} must be smaller than {num_players=}"
            )

        max_players = self.base_deck.total() // 2 - 1

        if num_players_alive > max_players:
            raise ValueError(
                f"{num_players_alive=} cannot be greater than "
                f"{max_players=} with current deck"
            )

        ### PLAYERS ###

        self.num_players_alive: int = num_players_alive

        self.possible_agents: list = [f"agent_{n}" for n in range(num_players)]
        self.agents: list[AgentID]

        self.players: list[Player]

        ### GAME STATE ###

        self.n_turn: int
        self.turn: state.Any

        self.rewards: dict[AgentID, float]
        self._cumulative_rewards: dict[AgentID, float]

        self.infos: dict[AgentID, dict]
        self.truncations: dict[AgentID, bool]
        self.terminations: dict[AgentID, bool]

        ### OBSERVATION SPACE ###

        obs_parts = state.observation_space(
            num_players=len(self.possible_agents),
            max_card_count=max(self.base_deck.values()),
        )

        highs = []
        shape = 0
        for record in obs_parts:
            shape += record["size"]
            highs += [record["max"]] * record["size"]

        obs_space = spaces.Box(
            low=0, high=np.array(highs), shape=(shape,), dtype=np.int8
        )

        self.observation_spaces = dict.fromkeys(self.possible_agents, obs_space)

        ### ACTION SPACE ###

        self.acts: ActionSpace = action.action_space(
            num_players=len(self.possible_agents)
        )

        self.act_to_idx = {act: i for i, act in enumerate(self.acts)}

        act_space = spaces.Discrete(len(self.acts))
        self.action_spaces = dict.fromkeys(self.possible_agents, act_space)

    @property
    def agent_selection(self) -> AgentID:
        """Return the current agent whose turn it is."""
        return self.turn.player.id

    def observation_space(self, agent) -> spaces.Box:
        """Return the observation space for the given agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent) -> spaces.Discrete:
        """Return the action space for the given agent."""
        return self.action_spaces[agent]

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment to initial state."""

        if options:
            logger.warning(
                "Options passed to reset method, but not used: options=%s", options
            )

        # reset numpy random seed
        np.random.seed(seed)

        self.game_id += 1
        logger.debug(
            "Environment reset: game_id=%d, num_players_alive=%d, num_players=%d",
            self.game_id,
            self.num_players_alive,
            len(self.possible_agents),
        )

        self.deck = self.base_deck.copy()
        logger.debug("Initial deck: %s", {str(k): v for k, v in self.deck.items()})

        shuffled_agents = self.possible_agents.copy()
        np.random.shuffle(shuffled_agents)

        self.agents = sorted(shuffled_agents[: self.num_players_alive])

        logger.debug("Active agents: %s", self.agents)

        self.players = [
            Player(id=agent, _deck=self.deck) for agent in self.possible_agents
        ]

        for i, player in enumerate(self.players):
            player.next = self.players[(i + 1) % len(self.players)]

            if player.id in self.agents:
                logger.debug("Initializing alive %s", player.id)

                player.draw()
                player.draw()

            elif self.dead_draw:
                logger.debug("Dead draw for %s", player.id)

                draw(self.deck)
                draw(self.deck)

        agent_idx = np.random.randint(0, len(self.agents))
        agent = self.agents[agent_idx]

        self.n_turn = 1
        self.turn = state.Start(player=self.find_player(agent))
        logger.debug("Initial turn: %s", self.turn.player.id)

        self.rewards = dict.fromkeys(self.agents, 0.0)
        self._cumulative_rewards = dict.fromkeys(self.agents, 0.0)

        self.infos = {agent: {"observation_history": []} for agent in self.agents}
        self.truncations = dict.fromkeys(self.agents, False)
        self.terminations = dict.fromkeys(self.agents, False)

    def find_player(self, agent: AgentID) -> Player:
        """Retrieve player object by agent ID."""

        for player in self.players:
            if player.id == agent:
                return player
        raise ValueError(f"{agent} not found")

    def observe(self, agent: AgentID) -> np.ndarray:
        """Generate observation for the given agent."""

        turn = self.turn

        player = self.find_player(agent)

        unseen = self.deck.copy()
        for _, nxt in player.enum(skip_self=True):
            for card in nxt.cards.elements():
                unseen[card] += 1

        obs = [nxt.coins for _, nxt in player.enum()]
        obs += [unseen[card] for card in CARD]
        obs += [player.cards[card] for card in CARD]
        obs += [nxt.cards.total() for _, nxt in player.enum()]
        obs += state.observe_act(getattr(turn, "act", None), player)
        obs += state.observe_challenge(getattr(turn, "challenge", None), player)
        obs += state.observe_block(getattr(turn, "block", None), player)
        obs += state.observe_challenge(getattr(turn, "block_challenge", None), player)

        return np.array(obs, dtype=np.int8)

    def get_action_mask(self) -> np.ndarray:
        """Generate valid action mask for the current player."""

        action_mask = [False] * len(self.acts)

        actions = self.turn.action_mask()

        for act, tgt in actions:
            idx = self.act_to_idx[(act, tgt)]
            action_mask[idx] = True

        return np.array(action_mask, dtype=np.int8)

    def last(self, observe: bool = True):
        """Return observation, reward, and game state for the last player to act."""

        player = self.turn.player

        reward = self.rewards[player.id]

        termination = self.terminations[player.id]

        truncation = self.truncations[player.id]

        self.infos[player.id]["action_mask"] = self.get_action_mask()

        info = self.infos[player.id]

        if observe:
            observation = self.observe(player.id)

            if not termination and not any(info["action_mask"]):
                raise ValueError("No available actions")
        else:
            observation = None

        return (
            observation,
            reward,
            termination,
            truncation,
            info,
        )

    def step(self, action: int | None):
        """Execute an action and transition to the next game state."""

        if action is None:
            assert type(self.turn) is state.EndTurn

            agent = self.turn.player.id

            self.agents.remove(agent)

            del self.terminations[agent]
            del self.truncations[agent]
            del self.infos[agent]
            del self.rewards[agent]

            logger.debug("%s removed from game", agent)

        else:
            turn = self.turn
            player = turn.player

            action_mask = self.get_action_mask()

            if not action_mask[action]:
                logger.error(
                    "Invalid action %d for %s in state %s",
                    action,
                    player.id,
                    type(turn).__name__,
                )
                raise ValueError(f"Action {action} is not an allowed action")

            action_obj, tgt = self.acts[action]

            target = player
            while target and tgt:
                target = target.next
                tgt -= 1

            if target is None or not target.alive:
                raise RuntimeError(f"Invalid target for action {action_obj.name}")

            logger.debug(
                "Step: turn_type=%s, player=%s, action=%s, target=%s",
                type(turn).__name__,
                player.id,
                action_obj.name,
                target.id,
            )

            self.turn = self.turn.step(action_obj, target)
            logger.debug("New turn state: %s", type(self.turn).__name__)

        if type(self.turn) is state.EndTurn:
            # get longest observation_history
            if self.agents:
                dct = max(
                    (x["observation_history"] for x in self.infos.values()),
                    key=lambda x: len(x),
                )

                if len(dct) < self.n_turn:
                    logger.debug("Saving observations for all players")
                    for player in self.players:
                        if player.id not in self.agents:
                            continue
                        obs = self.observe(player.id)
                        self.infos[player.id]["observation_history"].append(obs)

            for player in self.players:
                player.challenge_passed = False
                player.block_passed = False
                player.block_challenge_passed = False

                if not player.alive and player.id in self.agents:
                    logger.debug("%s died", player.id)
                    self.terminations[player.id] = True
                    self.turn = replace(self.turn, _player=player)
                    return

            if len(self.agents) == 1:
                winner_id = self.agents[0]
                winner = self.find_player(winner_id)
                logger.debug(
                    "%s won with %d cards\n",
                    winner.id,
                    winner.cards.total(),
                )
                self.rewards[winner.id] = 1.0
                self._cumulative_rewards[winner.id] = 1.0
                self.terminations[winner.id] = True
                self.turn = replace(self.turn, _player=winner)
                return

            if self.agents:
                logger.debug(
                    "Turn %d completed, moving to %s\n",
                    self.n_turn,
                    self.turn.act.actor.next_alive.id,
                )
                self.n_turn += 1
                self.turn = state.Start(player=self.turn.act.actor.next_alive)
                return

            return

    def render(self) -> str | None:
        """Render the current game state as a string."""

        if self.render_mode is None:
            logger.warning(
                "You are calling render method without specifying any render mode."
            )
            return None

        if self.render_mode != "ansi":
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

        if len(self.agents) == 1:
            return f"GAME OVER - Winner is {self.agents[0]}"

        actions = self.turn.action_mask()

        available_actions = []
        for act, tgt in actions:
            if tgt == 0:
                available_actions.append(f"{act.name}")
            else:
                available_actions.append(f"{act.name}_{tgt}")

        out = (
            f"\nTurn {self.n_turn}"
            + f" - {type(self.turn).__name__}"
            + f" - {self.turn.player.id}\n"
        )

        out += "\nAvailable actions: {}\n".format(" ".join(available_actions))

        players = [
            {
                "id": str(player.id),
                "cards": " ".join(str(card)[:3] for card in player.cards.elements()),
                "coins": f"{player.coins}$",
            }
            for player in self.players
        ]

        player_table = to_markdown(players)
        out += f"\n{player_table}\n"

        return out

    def close(self):
        """Clean up environment resources."""


def env(*args, **kwargs):
    """PettingZoo Coup game environment for RL training (equal to raw_env)."""
    return raw_env(*args, **kwargs)
