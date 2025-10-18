import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from coup_pettingzoo.env.card import CARD, draw

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Player:
    id: int
    _deck: Counter[CARD]
    coins: int = 2
    cards: Counter[CARD] = field(default_factory=Counter)

    block_passed: bool = False
    challenge_passed: bool = False
    block_challenge_passed: bool = False

    @property
    def alive(self) -> bool:
        return self.cards.total() > 0

    def draw(self):
        card = draw(self._deck)
        self.cards[card] += 1
        logger.debug(
            "Player %d drew %s; total cards=%d", self.id, str(card), self.cards.total()
        )

    def putback(self, card: CARD):
        assert card in self.cards

        self.cards[card] -= 1
        if not self.cards[card]:
            del self.cards[card]

        self._deck[card] += 1

        logger.debug(
            "Player %d put back %s; deck count=%d", self.id, str(card), self._deck[card]
        )

    def lose(self, card: CARD):
        assert card in self.cards

        self.cards[card] -= 1
        if not self.cards[card]:
            del self.cards[card]

        logger.debug(
            "Player %d loses %s; total cards=%d",
            self.id,
            str(card),
            self.cards.total(),
        )


def enum(
    players: list[Player],
    pos: int,
    alive: bool = False,
    skip_self: bool = False,
) -> Iterator[tuple[int, Player]]:
    from_player = players[pos + skip_self :] + players[:pos]

    for i, player in enumerate(from_player, start=skip_self):
        if not alive or player.alive:
            yield i, player


def nxt(players: list[Player], pos: int) -> Player:
    from_player = players[pos + 1 :] + players[:pos]

    for player in from_player:
        if player.alive:
            return player

    raise RuntimeError("No next alive player")
