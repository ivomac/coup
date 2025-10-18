"""Card types and deck management."""

import logging
from collections import Counter
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class CARD(Enum):
    """Card types in the Coup game."""

    AMBASSADOR = "Ambassador"
    ASSASSIN = "Assassin"
    CAPTAIN = "Captain"
    CONTESSA = "Contessa"
    DUKE = "Duke"

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


def draw(deck: Counter[CARD]) -> CARD:
    """Draw a random card from a deck."""
    cards = list(deck.elements())
    idx = np.random.randint(0, len(cards) - 1)
    card = cards[idx]

    logger.debug("Deck draw: total_cards=%d", len(cards))

    deck[card] -= 1
    if not deck[card]:
        del deck[card]

    logger.debug("Drew card: %s", str(card))
    return card
