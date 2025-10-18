"""Player representation and management."""

import logging
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NewType

from pettingzoo_coup.env.card import CARD, draw

logger = logging.getLogger(__name__)


AgentID = NewType("AgentID", str)


class PlayerError(Exception):
    """Raised when an invalid operation is attempted on a player."""


@dataclass(slots=True)
class Player:
    """Represents a single player in the game with cards and coins."""

    id: AgentID
    _deck: Counter[CARD]
    coins: int = 2
    cards: Counter[CARD] = field(default_factory=Counter)

    block_passed: bool = False
    challenge_passed: bool = False
    block_challenge_passed: bool = False

    next: "Player | None" = None

    @property
    def alive(self) -> bool:
        """Check if player is still in the game."""
        return self.cards.total() > 0

    @property
    def next_alive(self) -> "Player":
        """Return the next alive player in turn order."""
        player = self.next
        while player is not None and player != self:
            if player.alive:
                return player
            player = player.next
        raise PlayerError("No other alive players found")

    def draw(self):
        """Draw a card from the deck."""
        card = draw(self._deck)
        self.cards[card] += 1
        logger.debug(
            "%s drew %s; total cards=%d", self.id, str(card), self.cards.total()
        )

    def putback(self, card: CARD):
        """Return a card to the deck."""
        assert card in self.cards

        self.cards[card] -= 1
        if not self.cards[card]:
            del self.cards[card]

        self._deck[card] += 1

        logger.debug(
            "%s put back %s; deck count=%d", self.id, str(card), self._deck[card]
        )

    def lose(self, card: CARD):
        """Remove a card from player's hand permanently."""
        assert card in self.cards

        self.cards[card] -= 1
        if not self.cards[card]:
            del self.cards[card]

        logger.debug(
            "%s loses %s; total cards=%d",
            self.id,
            str(card),
            self.cards.total(),
        )

    def enum(
        self,
        skip_dead: bool = False,
        skip_self: bool = False,
    ) -> Iterator[tuple[int, "Player"]]:
        """Enumerate players in turn order with their relative indices."""
        if not skip_self and (not skip_dead or self.alive):
            yield 0, self

        i = 1
        player = self.next
        while player is not None and player != self:
            if not skip_dead or player.alive:
                yield i, player
            i += 1
            player = player.next
