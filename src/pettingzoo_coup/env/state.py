"""Game state transitions and state classes for the Coup game."""

import logging
from dataclasses import dataclass, replace
from typing import Literal

from pettingzoo_coup.env.action import ACT, ACTION, BLOCK_ACTION, START_ACTION, Action
from pettingzoo_coup.env.card import CARD
from pettingzoo_coup.env.player import Player

logger = logging.getLogger(__name__)


class InvalidState(Exception):
    """Raised when an invalid state is reached for debugging purposes."""


def lose_card_mask(player: Player) -> list[tuple[Action, int]]:
    """Generate valid card loss actions for a player."""

    actions = []

    for action in ACTION:
        if action.type == ACT.LOSE and player.cards[action.card]:
            actions.append((action, 0))

    return actions


def block_mask(act: Action) -> list[tuple[Action, int]]:
    """Generate valid block actions against the given action."""

    actions = []

    for action in ACTION:
        if action.name in act.block:
            actions.append((action, 0))

    return actions


def challenge_mask() -> list[tuple[Action, int]]:
    """Generate valid challenge actions."""

    actions = []

    for action in ACTION:
        if action.type == ACT.CHALLENGE:
            actions.append((action, 0))

    return actions


@dataclass(frozen=True, slots=True)
class ActInfo:
    """Information about an action being performed by a player."""

    actor: Player
    action: Action
    target: Player


def observe_act(act: ActInfo | None, player: Player):
    """Generate observation vector for an action."""

    if act:
        vec = [act.action == action for action in START_ACTION]  # idx
        vec += [act.actor == nxt for _, nxt in player.enum()]  # actor
        vec += [act.target == nxt for _, nxt in player.enum()]  # target

    else:
        vec = [False for _ in START_ACTION]
        vec += [False for _ in player.enum()]
        vec += [False for _ in player.enum()]

    return vec


def resolve_steal(act: ActInfo):
    """Execute a steal action, transferring coins between players."""

    amount = min(2, act.target.coins)

    act.actor.coins += amount
    act.target.coins -= amount

    logger.debug(
        "Steal resolved: %s steals %d coins from target=%s",
        act.actor.id,
        amount,
        act.target.id,
    )
    return


@dataclass(frozen=True, slots=True)
class ChallengeInfo:
    """Information about a challenge to an action or block."""

    type: Literal["action", "block"]
    challenger: Player
    loser: Player

    @property
    def failed(self) -> bool:
        return self.loser == self.challenger


def observe_challenge(chl: ChallengeInfo | None, player: Player):
    """Generate observation vector for a challenge."""

    if chl:
        if chl.type == "action":
            vec = [nxt.challenge_passed for _, nxt in player.enum()]

        elif chl.type == "block":
            vec = [nxt.block_challenge_passed for _, nxt in player.enum()]

        else:
            raise InvalidState("Invalid challenge type")

        vec += [chl.challenger == nxt for _, nxt in player.enum()]
        vec += [chl.loser == nxt for _, nxt in player.enum()]

    else:
        vec = [False for _ in player.enum()]
        vec += [False for _ in player.enum()]
        vec += [False for _ in player.enum()]

    return vec


@dataclass(frozen=True, slots=True)
class BlockInfo:
    """Information about a block attempt against an action."""

    blocker: Player
    action: Action


def observe_block(blk: BlockInfo | None, player: Player):
    """Generate observation vector for a block."""

    if blk:
        vec = [blk.action == action for action in BLOCK_ACTION]
        vec += [nxt.block_passed for _, nxt in player.enum()]
        vec += [blk.blocker == nxt for _, nxt in player.enum()]

    else:
        vec = [False for _ in range(len(BLOCK_ACTION))]
        vec += [False for _ in player.enum()]
        vec += [False for _ in player.enum()]

    return vec


@dataclass(frozen=True, slots=True)
class Start:
    """Initial state where a player selects an action to perform."""

    player: Player

    def action_mask(self) -> list[tuple[Action, int]]:
        actions = []

        player = self.player

        if player.coins >= 7:
            for i, _ in player.enum(skip_dead=True, skip_self=True):
                actions.append((ACTION.COUP, i))

            if player.coins >= 10:
                # must coup
                return actions

        for i, other in player.enum(skip_dead=True, skip_self=True):
            if player.coins >= 3:
                actions.append((ACTION.ASSASSINATE, i))
            if other.coins > 0:
                actions.append((ACTION.STEAL, i))

        for action in ACTION:
            if action.type == ACT.SELF:
                actions.append((action, 0))

        return actions

    def step(self, action: Action, target: Player) -> "Any":
        act = ActInfo(actor=self.player, action=action, target=target)

        logger.debug(
            "Start state: actor=%s action=%s target=%s",
            self.player.id,
            action.name,
            target.id,
        )

        if act.action == ACTION.INCOME:
            act.actor.coins += 1

            logger.debug(
                "Income: %s gains 1 coin, now has %d",
                act.actor.id,
                act.actor.coins,
            )

            return EndTurn(act=act)

        if act.action == ACTION.FOREIGN_AID:
            logger.debug("Foreign Aid: moving to block phase")

            return ForeignAidBlock(player=self.player.next_alive, act=act)

        if act.action == ACTION.COUP:
            act.actor.coins -= 7

            logger.debug(
                "Coup: %s pays 7 coins, now has %d",
                act.actor.id,
                act.actor.coins,
            )

            return ActionResolve(player=act.target, act=act)

        if act.action == ACTION.ASSASSINATE:
            act.actor.coins -= 3

            logger.debug(
                "Assassinate: %s pays 3 coins, now has %d",
                act.actor.id,
                act.actor.coins,
            )

        if act.action.card:
            logger.debug("Card-based action: moving to challenge phase")

            return Challenge(player=self.player.next_alive, act=act)

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class Challenge:
    """State where players can challenge an action."""

    player: Player
    act: ActInfo

    def action_mask(self) -> list[tuple[Action, int]]:
        return challenge_mask()

    def step(self, action: Action, *_) -> "Any":
        player = self.player
        logger.debug("Challenge state: player=%s action=%s", player.id, action.name)

        if action == ACTION.CHALLENGE_PASS:
            player.challenge_passed = True
            logger.debug("Challenge passed by %s", player.id)

            if self.player.next_alive == self.act.actor:
                if self.act.action == ACTION.EXCHANGE:
                    logger.debug("Exchange action proceeding")
                    self.act.actor.draw()
                    self.act.actor.draw()
                    return ExchangeResolve(player=self.act.actor, act=self.act)

                if self.act.action == ACTION.TAX:
                    self.act.actor.coins += 3
                    logger.debug(
                        "Tax action: %s gains 3 coins, now has %d",
                        self.act.actor.id,
                        self.act.actor.coins,
                    )
                    return EndTurn(act=self.act)

                if self.act.action.block:
                    logger.debug("Moving to target block phase")
                    return TargetBlock(player=self.act.target, act=self.act)

            else:
                logger.debug("Moving challenge to next player")

                return replace(self, player=self.player.next_alive)

        if action == ACTION.CHALLENGE_CALL:
            logger.debug("Challenge called by %s", player.id)

            actor = self.act.actor
            card = self.act.action.card

            assert card

            if actor.cards[card]:
                loser = player

                logger.debug("Challenge failed: actor has the card, challenger loses")
                actor.putback(card)
                actor.draw()
            else:
                loser = actor
                logger.debug(
                    "Challenge succeeded: actor doesn't have the card, actor loses"
                )

            chl = ChallengeInfo(type="action", challenger=self.player, loser=loser)

            logger.debug(
                "Challenge result: challenger=%s, loser=%s",
                chl.challenger.id,
                chl.loser.id,
            )

            return ChallengeResolve(player=loser, act=self.act, challenge=chl)

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class ChallengeResolve:
    """State where challenge loser loses a card."""

    player: Player
    act: ActInfo
    challenge: ChallengeInfo

    def action_mask(self) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "Any":
        card = action.card
        assert card

        logger.debug("Challenge resolve: player=%s loses %s", self.player.id, card.name)

        self.player.lose(card)

        if self.challenge.loser == self.act.actor:
            logger.debug("Challenge loser was actor, ending turn")

            return EndTurn(act=self.act, challenge=self.challenge)

        if self.act.action == ACTION.TAX:
            logger.debug("Tax action proceeding after challenge")

            self.act.actor.coins += 3

            logger.debug(
                "Tax action: %s gains 3 coins, now has %d",
                self.act.actor.id,
                self.act.actor.coins,
            )

            return EndTurn(act=self.act, challenge=self.challenge)

        if self.act.action == ACTION.EXCHANGE:
            logger.debug("Exchange action proceeding after challenge")

            self.act.actor.draw()
            self.act.actor.draw()

            return ExchangeResolve(player=self.act.actor, act=self.act)

        if self.act.action == ACTION.STEAL and not self.act.target.alive:
            logger.debug("Steal action with dead target, resolving steal")

            resolve_steal(self.act)

            return EndTurn(act=self.act, challenge=self.challenge)

        if self.act.action == ACTION.ASSASSINATE and not self.act.target.alive:
            logger.debug("Assassinate action with dead target, ending turn")

            return EndTurn(act=self.act, challenge=self.challenge)

        if self.act.action.block:
            logger.debug("Moving to target block phase after challenge")

            return TargetBlock(player=self.act.target, act=self.act)

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class ForeignAidBlock:
    """State where players can block a foreign aid action."""

    player: Player
    act: ActInfo

    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return block_mask(self.act.action)

    def step(self, action: Action, *_) -> "Any":
        player = self.player
        logger.debug("ForeignAidBlock: player=%s action=%s", player.id, action.name)

        if action == ACTION.BLOCK_PASS:
            player.block_passed = True
            logger.debug("Block passed by %s", player.id)

            if self.player.next_alive == self.act.actor:
                self.act.actor.coins += 2

                logger.debug(
                    "Foreign Aid successful: %s gains 2 coins, now has %d",
                    self.act.actor.id,
                    self.act.actor.coins,
                )

                return EndTurn(act=self.act, challenge=self.challenge)

            logger.debug("Moving block to next player")

            return replace(self, player=self.player.next_alive)

        if action == ACTION.BLOCK_FOREIGN_AID:
            logger.debug("Foreign Aid blocked by %s", player.id)

            block = BlockInfo(blocker=self.player, action=ACTION.BLOCK_FOREIGN_AID)
            return BlockChallenge(
                player=self.player, act=self.act, challenge=self.challenge, block=block
            )

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class TargetBlock:
    """State where the target player can block an action."""

    player: Player
    act: ActInfo

    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return block_mask(self.act.action)

    def step(self, action: Action, *_) -> "Any":
        player = self.player

        logger.debug("TargetBlock: player=%s action=%s", player.id, action.name)

        if action == ACTION.BLOCK_PASS:
            player.block_passed = True

            logger.debug("Block passed by %s", player.id)

            if self.act.action == ACTION.STEAL:
                logger.debug("Steal action proceeding after block pass")

                resolve_steal(self.act)
                return EndTurn(act=self.act, challenge=self.challenge)
            if self.act.action == ACTION.ASSASSINATE:
                logger.debug("Assassinate action proceeding after block pass")

                return ActionResolve(
                    player=self.act.target, act=self.act, challenge=self.challenge
                )

        else:
            logger.debug("Action blocked by %s with %s", player.id, action.name)

            block = BlockInfo(blocker=self.player, action=action)
            return BlockChallenge(
                player=self.player, act=self.act, challenge=self.challenge, block=block
            )

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class BlockChallenge:
    """State where players can challenge a block."""

    player: Player

    block: BlockInfo

    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return challenge_mask()

    def step(self, action: Action, *_) -> "Any":
        player = self.player

        logger.debug("BlockChallenge: player=%s action=%s", player.id, action.name)

        if action == ACTION.CHALLENGE_PASS:
            player.challenge_passed = True

            logger.debug("Block challenge passed by %s", player.id)

            if self.player.next_alive == self.act.actor:
                logger.debug("Block challenge phase complete")

                return EndTurn(act=self.act, challenge=self.challenge, block=self.block)

            logger.debug("Moving block challenge to next player")

            return replace(self, player=self.player.next_alive)

        if action == ACTION.CHALLENGE_CALL:
            logger.debug("Block challenge called by %s", player.id)

            blocker = self.block.blocker
            card = self.block.action.card
            assert card

            if blocker.cards[card]:
                loser = player

                logger.debug(
                    "Block challenge failed: blocker has the card, challenger loses"
                )

                blocker.putback(card)
                blocker.draw()
            else:
                loser = blocker

                logger.debug(
                    "Block challenge succeeded: blocker doesn't have the card, blocker loses"
                )

            chl = ChallengeInfo(type="block", challenger=self.player, loser=loser)

            logger.debug(
                "Block challenge result: challenger=%s, loser=%s",
                chl.challenger.id,
                chl.loser.id,
            )

            return BlockChallengeResolve(
                player=loser,
                act=self.act,
                challenge=self.challenge,
                block=self.block,
                block_challenge=chl,
            )

        raise InvalidState("Action not handled for this state")


@dataclass(frozen=True, slots=True)
class BlockChallengeResolve:
    """State where block challenge loser loses a card."""

    player: Player

    block: BlockInfo
    block_challenge: ChallengeInfo

    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "Any":
        card = action.card
        assert card

        logger.debug(
            "BlockChallengeResolve: player=%s loses %s", self.player.id, card.name
        )

        self.player.lose(card)

        if self.block_challenge.loser == self.block_challenge.challenger:
            logger.debug("Block challenge loser was challenger, ending turn")

            return self.toEndTurn()

        if self.act.action == ACTION.FOREIGN_AID:
            self.act.actor.coins += 2
            logger.debug(
                "Foreign Aid successful after block challenge: %s gains 2 coins, now has %d",
                self.act.actor.id,
                self.act.actor.coins,
            )

            return self.toEndTurn()

        if self.act.action == ACTION.STEAL:
            logger.debug("Steal action proceeding after block challenge")

            resolve_steal(self.act)
            return self.toEndTurn()

        if self.act.action == ACTION.ASSASSINATE:
            if not self.act.target.alive:
                logger.debug("Assassinate action with dead target, ending turn")

                return self.toEndTurn()

            logger.debug("Assassinate action proceeding after block challenge")

            return ActionResolve(
                player=self.act.target,
                act=self.act,
                challenge=self.challenge,
                block=self.block,
                block_challenge=self.block_challenge,
            )

        raise InvalidState("Action not handled for this state")

    def toEndTurn(self) -> "EndTurn":
        return EndTurn(
            act=self.act,
            challenge=self.challenge,
            block=self.block,
            block_challenge=self.block_challenge,
        )


@dataclass(frozen=True, slots=True)
class ActionResolve:
    """State where action target loses a card to resolve assassination."""

    player: Player

    act: ActInfo
    challenge: ChallengeInfo | None = None

    block: BlockInfo | None = None
    block_challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "EndTurn":
        card = action.card
        assert card

        logger.debug("ActionResolve: player=%s loses %s", self.player.id, card.name)

        self.player.lose(card)

        logger.debug("Action resolved, ending turn")

        return EndTurn(
            act=self.act,
            challenge=self.challenge,
            block=self.block,
            block_challenge=self.block_challenge,
        )


@dataclass(frozen=True, slots=True)
class ExchangeResolve:
    """State where player returns the first of two drawn cards."""

    player: Player
    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "ExchangeTwoResolve":
        card = action.card
        assert card

        logger.debug(
            "ExchangeResolve: player=%s puts back %s", self.player.id, card.name
        )

        self.player.putback(card)

        logger.debug("First exchange card resolved, moving to second exchange")

        return ExchangeTwoResolve(
            player=self.player, act=self.act, challenge=self.challenge
        )


@dataclass(frozen=True, slots=True)
class ExchangeTwoResolve:
    """State where player returns the second of two drawn cards."""

    player: Player
    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "EndTurn":
        card = action.card
        assert card

        logger.debug(
            "ExchangeTwoResolve: player=%s puts back %s", self.player.id, card.name
        )

        self.player.putback(card)

        logger.debug("Second exchange card resolved, ending turn")

        return EndTurn(act=self.act, challenge=self.challenge)


@dataclass(frozen=True, slots=True)
class EndTurn:
    """State marking the end of a turn, before moving to the next player."""

    act: ActInfo
    challenge: ChallengeInfo | None = None

    block: BlockInfo | None = None
    block_challenge: ChallengeInfo | None = None

    _player: Player | None = None

    @property
    def player(self) -> Player:
        if self._player:
            return self._player
        raise InvalidState("EndTurn is current state but no player specified")

    def step(self, *_) -> "EndTurn":
        raise NotImplementedError

    def action_mask(self) -> list[tuple[Action, int]]:
        return []


@dataclass(frozen=True, slots=True)
class GameOver:
    """Terminal state indicating the game has ended."""

    @property
    def player(self):
        raise NotImplementedError

    def step(self, *_) -> "GameOver":
        return self

    def action_mask(self) -> list[tuple[Action, int]]:
        return []


Any = (
    Start
    | Challenge
    | ChallengeResolve
    | ForeignAidBlock
    | TargetBlock
    | BlockChallenge
    | BlockChallengeResolve
    | ExchangeResolve
    | ExchangeTwoResolve
    | ActionResolve
    | EndTurn
    | GameOver
)


def observation_space(num_players: int, max_card_count: int) -> list[dict]:
    return [
        {
            "type": "Counts",
            "desc": "Coins of each player",
            "max": 12,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "Counts",
            "desc": "Unseen card totals",
            "max": max_card_count,
            "size": len(CARD),
            "size_desc": "# Card types",
            "scope": "Private",
        },
        {
            "type": "Counts",
            "desc": "Player's cards by type",
            "max": max_card_count,
            "size": len(CARD),
            "size_desc": "# Card types",
            "scope": "Private",
        },
        {
            "type": "Counts",
            "desc": "Card totals per player",
            "max": 4,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Action",
            "max": 1,
            "size": len(START_ACTION),
            "size_desc": "# Start Actions",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Actor",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Action's target",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "Indicator",
            "desc": "Challenge passed",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Challenger",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Challenge loser",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Block action",
            "max": 1,
            "size": len(BLOCK_ACTION),
            "size_desc": "# Block Actions",
            "scope": "Public",
        },
        {
            "type": "Indicator",
            "desc": "Block passed",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Blocker",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "Indicator",
            "desc": "Block challenge passed",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Block challenger",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
        {
            "type": "One-hot",
            "desc": "Block challenge loser",
            "max": 1,
            "size": num_players,
            "size_desc": "# Players",
            "scope": "Public",
        },
    ]
