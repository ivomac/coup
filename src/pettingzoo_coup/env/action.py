"""Action types and definitions."""

from dataclasses import dataclass, field
from enum import Enum, auto

from pettingzoo_coup.env.card import CARD


class ACT(Enum):
    """Action type categories."""

    SELF = auto()
    TARGET = auto()
    BLOCK = auto()
    CHALLENGE = auto()
    LOSE = auto()

    def __repr__(self):
        return f"ACT.{self.name}"

    def __str__(self):
        return repr(self)


@dataclass(frozen=True, slots=True)
class Action:
    """Base class for all actions in the game."""

    type: ACT
    name: str
    card: CARD | None = None
    block: tuple = field(default_factory=tuple)

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


@dataclass(frozen=True, slots=True)
class ActionTypes:
    """Container for all predefined action instances in the game."""

    LOSE_AMBASSADOR: Action = Action(ACT.LOSE, "LOSE_AMBASSADOR", card=CARD.AMBASSADOR)
    LOSE_ASSASSIN: Action = Action(ACT.LOSE, "LOSE_ASSASSIN", card=CARD.ASSASSIN)
    LOSE_CAPTAIN: Action = Action(ACT.LOSE, "LOSE_CAPTAIN", card=CARD.CAPTAIN)
    LOSE_CONTESSA: Action = Action(ACT.LOSE, "LOSE_CONTESSA", card=CARD.CONTESSA)
    LOSE_DUKE: Action = Action(ACT.LOSE, "LOSE_DUKE", card=CARD.DUKE)

    CHALLENGE_PASS: Action = Action(ACT.CHALLENGE, "CHALLENGE_PASS")
    CHALLENGE_CALL: Action = Action(ACT.CHALLENGE, "CHALLENGE_CALL")

    BLOCK_PASS: Action = Action(ACT.BLOCK, "BLOCK_PASS")
    BLOCK_ASSASSINATE: Action = Action(
        ACT.BLOCK, "BLOCK_ASSASSINATE", card=CARD.CONTESSA
    )
    BLOCK_FOREIGN_AID: Action = Action(ACT.BLOCK, "BLOCK_FOREIGN_AID", card=CARD.DUKE)
    BLOCK_STEAL_AMB: Action = Action(ACT.BLOCK, "BLOCK_STEAL_AMB", card=CARD.AMBASSADOR)
    BLOCK_STEAL_CAP: Action = Action(ACT.BLOCK, "BLOCK_STEAL_CAP", card=CARD.CAPTAIN)

    EXCHANGE: Action = Action(ACT.SELF, "EXCHANGE", card=CARD.AMBASSADOR)
    FOREIGN_AID: Action = Action(
        ACT.SELF,
        "FOREIGN_AID",
        block=(
            "BLOCK_PASS",
            "BLOCK_FOREIGN_AID",
        ),
    )
    INCOME: Action = Action(ACT.SELF, "INCOME")
    TAX: Action = Action(ACT.SELF, "TAX", card=CARD.DUKE)

    ASSASSINATE: Action = Action(
        ACT.TARGET,
        "ASSASSINATE",
        card=CARD.ASSASSIN,
        block=("BLOCK_PASS", "BLOCK_ASSASSINATE"),
    )
    COUP: Action = Action(ACT.TARGET, "COUP")
    STEAL: Action = Action(
        ACT.TARGET,
        "STEAL",
        card=CARD.CAPTAIN,
        block=("BLOCK_PASS", "BLOCK_STEAL_AMB", "BLOCK_STEAL_CAP"),
    )

    def __len__(self):
        return len(self.__dataclass_fields__)

    def __iter__(self):
        return (getattr(self, name) for name in self.__dataclass_fields__)


ACTION: ActionTypes = ActionTypes()

START_ACTION: list[Action] = [
    action for action in ACTION if action.type in {ACT.SELF, ACT.TARGET}
]
BLOCK_ACTION: list[Action] = [action for action in ACTION if action.type == ACT.BLOCK]


def action_space(num_players: int) -> list[tuple[Action, int]]:
    acts = []

    for action in ACTION:
        if action.type != ACT.TARGET:
            acts.append((action, 0))
        else:
            for target in range(1, num_players):
                acts.append((action, target))

    return acts
