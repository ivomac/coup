import logging
from dataclasses import dataclass, replace

from coup_pettingzoo.env.action import ACT, ACTION, Action
from coup_pettingzoo.env.player import Player, enum, nxt

logger = logging.getLogger(__name__)


class InvalidState(Exception):
    pass


def lose_card_mask(player: Player) -> list[tuple[Action, int]]:
    actions = []

    for action in ACTION.iter():
        if action.type == ACT.LOSE and player.cards[action.card]:
            actions.append((action, 0))

    return actions


def block_mask(act: Action) -> list[tuple[Action, int]]:
    actions = []

    for action in ACTION.iter():
        if action.name in act.block:
            actions.append((action, 0))

    return actions


def challenge_mask() -> list[tuple[Action, int]]:
    actions = []

    for action in ACTION.iter():
        if action.type == ACT.CHALLENGE:
            actions.append((action, 0))

    return actions


@dataclass(frozen=True, slots=True)
class ActInfo:
    actor: Player
    action: Action
    target: Player


@dataclass(frozen=True, slots=True)
class BlockInfo:
    blocker: Player
    action: Action


@dataclass(frozen=True, slots=True)
class ChallengeInfo:
    challenger: Player
    loser: Player

    @property
    def failed(self) -> bool:
        return self.loser == self.challenger


def resolve_steal(act: ActInfo):
    amount = min(2, act.target.coins)
    logger.debug(
        "Steal resolved: actor=%d steals %d coins from target=%d",
        act.actor.id,
        amount,
        act.target.id,
    )
    act.actor.coins += amount
    act.target.coins -= amount
    logger.debug(
        "After steal: actor coins=%d, target coins=%d",
        act.actor.coins,
        act.target.coins,
    )
    return


@dataclass(frozen=True, slots=True)
class Start:
    player: Player

    def action_mask(self, players: list[Player]) -> list[tuple[Action, int]]:
        actions = []

        player = self.player

        if player.coins >= 7:
            for i, _ in enum(players, player.id, alive=True, skip_self=True):
                actions.append((ACTION.COUP, i))

            if player.coins >= 10:
                # must coup
                return actions

        for i, other in enum(players, player.id, alive=True, skip_self=True):
            if player.coins >= 3:
                actions.append((ACTION.ASSASSINATE, i))
            if other.coins > 0:
                actions.append((ACTION.STEAL, i))

        for action in ACTION.iter():
            if action.type == ACT.SELF:
                actions.append((action, 0))

        return actions

    def step(self, action: Action, target: Player, players: list[Player]) -> "Any":
        act = ActInfo(actor=self.player, action=action, target=target)
        logger.debug(
            "Start state: player=%d action=%s target=%d",
            self.player.id,
            action.name,
            target.id,
        )

        if act.action == ACTION.INCOME:
            act.actor.coins += 1
            logger.debug(
                "Income: player %d gains 1 coin, now has %d",
                act.actor.id,
                act.actor.coins,
            )
            return EndTurn(act=act)

        if act.action == ACTION.FOREIGN_AID:
            logger.debug("Foreign Aid: moving to block phase")
            return ForeignAidBlock(player=nxt(players, self.player.id), act=act)

        if act.action == ACTION.COUP:
            act.actor.coins -= 7
            logger.debug(
                "Coup: player %d pays 7 coins, now has %d",
                act.actor.id,
                act.actor.coins,
            )
            return ActionResolve(player=act.target, act=act)

        if act.action == ACTION.ASSASSINATE:
            act.actor.coins -= 3
            logger.debug(
                "Assassinate: player %d pays 3 coins, now has %d",
                act.actor.id,
                act.actor.coins,
            )

        if act.action.card:
            logger.debug("Card-based action: moving to challenge phase")
            return Challenge(player=nxt(players, self.player.id), act=act)

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class Challenge:
    player: Player
    act: ActInfo

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return challenge_mask()

    def step(self, action: Action, _: Player, players: list[Player]) -> "Any":
        player = self.player
        logger.debug("Challenge state: player=%d action=%s", player.id, action.name)

        if action == ACTION.CHALLENGE_PASS:
            player.challenge_passed = True
            logger.debug("Challenge passed by player %d", player.id)

            if nxt == self.act.actor:
                if self.act.action == ACTION.EXCHANGE:
                    logger.debug("Exchange action proceeding")
                    self.act.actor.draw()
                    self.act.actor.draw()
                    return ExchangeResolve(player=self.act.actor, act=self.act)

                if self.act.action == ACTION.TAX:
                    self.act.actor.coins += 3
                    logger.debug(
                        "Tax action: player %d gains 3 coins, now has %d",
                        self.act.actor.id,
                        self.act.actor.coins,
                    )
                    return EndTurn(act=self.act)

                if self.act.action.block:
                    logger.debug("Moving to target block phase")
                    return TargetBlock(player=self.act.target, act=self.act)

            else:
                logger.debug("Moving challenge to next player")
                return replace(self, player=nxt(players, self.player.id))

        if action == ACTION.CHALLENGE_CALL:
            logger.debug("Challenge called by player %d", player.id)
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

            chl = ChallengeInfo(challenger=self.player, loser=loser)
            logger.debug(
                "Challenge result: challenger=%d, loser=%d",
                chl.challenger.id,
                chl.loser.id,
            )
            return ChallengeResolve(player=loser, act=self.act, challenge=chl)

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class ChallengeResolve:
    player: Player
    act: ActInfo
    challenge: ChallengeInfo

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_):
        card = action.card
        assert card
        logger.debug("Challenge resolve: player=%d loses %s", self.player.id, card.name)

        self.player.lose(card)

        if self.challenge.loser == self.act.actor:
            logger.debug("Challenge loser was actor, ending turn")
            return EndTurn(act=self.act, challenge=self.challenge)

        if self.act.action == ACTION.TAX:
            logger.debug("Tax action proceeding after challenge")
            self.act.actor.coins += 3
            logger.debug(
                "Tax action: player %d gains 3 coins, now has %d",
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

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class ForeignAidBlock:
    player: Player
    act: ActInfo

    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return block_mask(self.act.action)

    def step(self, action: Action, _: Player, players: list[Player]) -> "Any":
        player = self.player
        logger.debug("ForeignAidBlock: player=%d action=%s", player.id, action.name)

        if action == ACTION.BLOCK_PASS:
            player.block_passed = True
            logger.debug("Block passed by player %d", player.id)

            if nxt == self.act.actor:
                self.act.actor.coins += 2
                logger.debug(
                    "Foreign Aid successful: player %d gains 2 coins, now has %d",
                    self.act.actor.id,
                    self.act.actor.coins,
                )
                return EndTurn(act=self.act, challenge=self.challenge)
            logger.debug("Moving block to next player")
            return replace(self, player=nxt(players, self.player.id))

        if action == ACTION.BLOCK_FOREIGN_AID:
            logger.debug("Foreign Aid blocked by player %d", player.id)
            block = BlockInfo(blocker=self.player, action=ACTION.BLOCK_FOREIGN_AID)
            return BlockChallenge(
                player=self.player, act=self.act, challenge=self.challenge, block=block
            )

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class TargetBlock:
    player: Player
    act: ActInfo

    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return block_mask(self.act.action)

    def step(self, action: Action, *_) -> "Any":
        player = self.player
        logger.debug("TargetBlock: player=%d action=%s", player.id, action.name)

        if action == ACTION.BLOCK_PASS:
            player.block_passed = True
            logger.debug("Block passed by player %d", player.id)

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
            logger.debug("Action blocked by player %d with %s", player.id, action.name)
            block = BlockInfo(blocker=self.player, action=action)
            return BlockChallenge(
                player=self.player, act=self.act, challenge=self.challenge, block=block
            )

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class BlockChallenge:
    player: Player

    block: BlockInfo

    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return challenge_mask()

    def step(self, action: Action, _: Player, players: list[Player]) -> "Any":
        player = self.player
        logger.debug("BlockChallenge: player=%d action=%s", player.id, action.name)

        if action == ACTION.CHALLENGE_PASS:
            player.challenge_passed = True
            logger.debug("Block challenge passed by player %d", player.id)

            if nxt == self.act.actor:
                logger.debug("Block challenge phase complete")
                return EndTurn(act=self.act, challenge=self.challenge, block=self.block)
            logger.debug("Moving block challenge to next player")
            return replace(self, player=nxt(players, self.player.id))

        if action == ACTION.CHALLENGE_CALL:
            logger.debug("Block challenge called by player %d", player.id)
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

            chl = ChallengeInfo(challenger=self.player, loser=loser)
            logger.debug(
                "Block challenge result: challenger=%d, loser=%d",
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

        raise InvalidState("Action not handled for this state.")


@dataclass(frozen=True, slots=True)
class BlockChallengeResolve:
    player: Player

    block: BlockInfo
    block_challenge: ChallengeInfo

    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "Any":
        card = action.card
        assert card
        logger.debug(
            "BlockChallengeResolve: player=%d loses %s", self.player.id, card.name
        )

        self.player.lose(card)

        if self.block_challenge.loser == self.block_challenge.challenger:
            logger.debug("Block challenge loser was challenger, ending turn")
            return self.toEndTurn()

        if self.act.action == ACTION.FOREIGN_AID:
            self.act.actor.coins += 2
            logger.debug(
                "Foreign Aid successful after block challenge: player %d gains 2 coins, now has %d",
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

        raise InvalidState("Action not handled for this state.")

    def toEndTurn(self) -> "EndTurn":
        return EndTurn(
            act=self.act,
            challenge=self.challenge,
            block=self.block,
            block_challenge=self.block_challenge,
        )


@dataclass(frozen=True, slots=True)
class ActionResolve:
    player: Player

    act: ActInfo
    challenge: ChallengeInfo | None = None

    block: BlockInfo | None = None
    block_challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "EndTurn":
        card = action.card
        assert card
        logger.debug("ActionResolve: player=%d loses %s", self.player.id, card.name)

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
    player: Player
    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "ExchangeTwoResolve":
        card = action.card
        assert card
        logger.debug(
            "ExchangeResolve: player=%d puts back %s", self.player.id, card.name
        )

        self.player.putback(card)

        logger.debug("First exchange card resolved, moving to second exchange")
        return ExchangeTwoResolve(
            player=self.player, act=self.act, challenge=self.challenge
        )


@dataclass(frozen=True, slots=True)
class ExchangeTwoResolve:
    player: Player
    act: ActInfo
    challenge: ChallengeInfo | None = None

    def action_mask(self, _: list[Player]) -> list[tuple[Action, int]]:
        return lose_card_mask(self.player)

    def step(self, action: Action, *_) -> "EndTurn":
        card = action.card
        assert card
        logger.debug(
            "ExchangeTwoResolve: player=%d puts back %s", self.player.id, card.name
        )

        self.player.putback(card)

        logger.debug("Second exchange card resolved, ending turn")
        return EndTurn(act=self.act, challenge=self.challenge)


@dataclass(frozen=True, slots=True)
class EndTurn:
    act: ActInfo
    challenge: ChallengeInfo | None = None

    block: BlockInfo | None = None
    block_challenge: ChallengeInfo | None = None

    @property
    def player(self):
        raise NotImplementedError

    def step(self, *_) -> "Any":
        raise NotImplementedError

    def action_mask(self, *_) -> list[tuple[Action, int]]:
        raise NotImplementedError


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
)
