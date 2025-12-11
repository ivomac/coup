"""Comprehensive tests for game mechanics and state transitions."""

import unittest
from collections import Counter

from pettingzoo_coup.env.action import ACTION
from pettingzoo_coup.env.card import CARD
from pettingzoo_coup.env.env import raw_env


class TestBasicActions(unittest.TestCase):
    """Test basic self-actions: Income, Tax, Foreign Aid, Exchange."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_income_action(self):
        """Test that Income action gives 1 coin and ends turn."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        initial_coins = player.coins

        income_idx = self.env.act_to_idx[(ACTION.INCOME, 0)]

        self.env.step(income_idx)

        self.assertEqual(player.coins, initial_coins + 1)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_tax_action_no_challenge(self):
        """Test Tax action when no one challenges."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.DUKE: 2})
        initial_coins = player.coins

        tax_idx = self.env.act_to_idx[(ACTION.TAX, 0)]

        self.env.step(tax_idx)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]

        self.env.step(challenge_pass_idx)
        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        self.env.step(challenge_pass_idx)

        self.assertEqual(player.coins, initial_coins + 3)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_foreign_aid_no_block(self):
        """Test Foreign Aid when no one blocks."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        initial_coins = player.coins

        foreign_aid_idx = self.env.act_to_idx[(ACTION.FOREIGN_AID, 0)]

        self.env.step(foreign_aid_idx)

        self.assertEqual(type(self.env.turn).__name__, "ForeignAidBlock")

        block_pass_idx = self.env.act_to_idx[(ACTION.BLOCK_PASS, 0)]

        self.env.step(block_pass_idx)
        self.assertEqual(type(self.env.turn).__name__, "ForeignAidBlock")

        self.env.step(block_pass_idx)

        self.assertEqual(player.coins, initial_coins + 2)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_foreign_aid_with_block(self):
        """Test Foreign Aid when someone blocks it."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        initial_coins = player.coins

        foreign_aid_idx = self.env.act_to_idx[(ACTION.FOREIGN_AID, 0)]

        self.env.step(foreign_aid_idx)

        self.assertEqual(type(self.env.turn).__name__, "ForeignAidBlock")

        block_idx = self.env.act_to_idx[(ACTION.BLOCK_FOREIGN_AID, 0)]
        self.env.step(block_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]

        self.env.step(challenge_pass_idx)
        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")

        self.env.step(challenge_pass_idx)

        self.assertEqual(player.coins, initial_coins)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestTargetedActions(unittest.TestCase):
    """Test targeted actions: Coup, Assassinate, Steal."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_coup_action(self):
        """Test Coup action (forces target to lose a card)."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 7

        target = player.next_alive
        target.cards = Counter({CARD.DUKE: 2})
        initial_target_cards = target.cards.total()

        coup_idx = self.env.act_to_idx[(ACTION.COUP, 1)]

        self.env.step(coup_idx)

        self.assertEqual(player.coins, 0)

        self.assertEqual(type(self.env.turn).__name__, "ActionResolve")
        self.assertEqual(self.env.turn.player.id, target.id)

        lose_duke_idx = self.env.act_to_idx[(ACTION.LOSE_DUKE, 0)]
        self.env.step(lose_duke_idx)

        self.assertEqual(target.cards.total(), initial_target_cards - 1)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_assassinate_no_block(self):
        """Test Assassinate action when not blocked."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 3
        player.cards = Counter({CARD.ASSASSIN: 2})

        target = player.next_alive
        target.cards = Counter({CARD.DUKE: 2})
        initial_target_cards = target.cards.total()

        assassinate_idx = self.env.act_to_idx[(ACTION.ASSASSINATE, 1)]

        self.env.step(assassinate_idx)

        self.assertEqual(player.coins, 0)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]

        self.env.step(challenge_pass_idx)

        self.env.step(challenge_pass_idx)

        self.assertEqual(type(self.env.turn).__name__, "TargetBlock")

        block_pass_idx = self.env.act_to_idx[(ACTION.BLOCK_PASS, 0)]
        self.env.step(block_pass_idx)

        self.assertEqual(type(self.env.turn).__name__, "ActionResolve")

        lose_duke_idx = self.env.act_to_idx[(ACTION.LOSE_DUKE, 0)]
        self.env.step(lose_duke_idx)

        self.assertEqual(target.cards.total(), initial_target_cards - 1)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_steal_no_block(self):
        """Test Steal action when not blocked."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.CAPTAIN: 2})
        initial_coins = player.coins

        target = player.next_alive
        target.coins = 5
        initial_target_coins = target.coins

        steal_idx = self.env.act_to_idx[(ACTION.STEAL, 1)]

        self.env.step(steal_idx)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]

        self.env.step(challenge_pass_idx)

        self.env.step(challenge_pass_idx)

        self.assertEqual(type(self.env.turn).__name__, "TargetBlock")

        block_pass_idx = self.env.act_to_idx[(ACTION.BLOCK_PASS, 0)]
        self.env.step(block_pass_idx)

        self.assertEqual(player.coins, initial_coins + 2)
        self.assertEqual(target.coins, initial_target_coins - 2)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestChallenges(unittest.TestCase):
    """Test challenge mechanics for actions."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_successful_challenge(self):
        """Test successful challenge (actor doesn't have the card)."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.AMBASSADOR: 2})
        initial_cards = player.cards.total()

        tax_idx = self.env.act_to_idx[(ACTION.TAX, 0)]

        self.env.step(tax_idx)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_call_idx = self.env.act_to_idx[(ACTION.CHALLENGE_CALL, 0)]
        self.env.step(challenge_call_idx)

        self.assertEqual(type(self.env.turn).__name__, "ChallengeResolve")
        self.assertEqual(self.env.turn.player.id, player.id)

        lose_amb_idx = self.env.act_to_idx[(ACTION.LOSE_AMBASSADOR, 0)]
        self.env.step(lose_amb_idx)

        self.assertEqual(player.cards.total(), initial_cards - 1)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_failed_challenge(self):
        """Test failed challenge (actor DOES have the card)."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.DUKE: 2})
        initial_coins = player.coins

        challenger = player.next_alive
        challenger.cards = Counter({CARD.AMBASSADOR: 2})
        initial_challenger_cards = challenger.cards.total()

        tax_idx = self.env.act_to_idx[(ACTION.TAX, 0)]

        self.env.step(tax_idx)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_call_idx = self.env.act_to_idx[(ACTION.CHALLENGE_CALL, 0)]
        self.env.step(challenge_call_idx)

        self.assertEqual(type(self.env.turn).__name__, "ChallengeResolve")
        self.assertEqual(self.env.turn.player.id, challenger.id)

        lose_amb_idx = self.env.act_to_idx[(ACTION.LOSE_AMBASSADOR, 0)]
        self.env.step(lose_amb_idx)

        self.assertEqual(challenger.cards.total(), initial_challenger_cards - 1)

        self.assertEqual(player.coins, initial_coins + 3)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestBlocks(unittest.TestCase):
    """Test block mechanics for targeted actions."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_assassinate_blocked(self):
        """Test Assassinate blocked by Contessa."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 3
        player.cards = Counter({CARD.ASSASSIN: 2})

        target = player.next_alive
        target.cards = Counter({CARD.CONTESSA: 2})
        initial_target_cards = target.cards.total()

        assassinate_idx = self.env.act_to_idx[(ACTION.ASSASSINATE, 1)]
        self.env.step(assassinate_idx)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        block_idx = self.env.act_to_idx[(ACTION.BLOCK_ASSASSINATE, 0)]
        self.env.step(block_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")

        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        self.assertEqual(target.cards.total(), initial_target_cards)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)

    def test_steal_blocked_by_captain(self):
        """Test Steal blocked by Captain."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.CAPTAIN: 2})
        initial_coins = player.coins

        target = player.next_alive
        target.cards = Counter({CARD.CAPTAIN: 2})
        target.coins = 5
        initial_target_coins = target.coins

        steal_idx = self.env.act_to_idx[(ACTION.STEAL, 1)]
        self.env.step(steal_idx)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        block_idx = self.env.act_to_idx[(ACTION.BLOCK_STEAL_CAP, 0)]
        self.env.step(block_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")

        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        self.assertEqual(player.coins, initial_coins)
        self.assertEqual(target.coins, initial_target_coins)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestBlockChallenges(unittest.TestCase):
    """Test block challenge mechanics."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_successful_block_challenge(self):
        """Test successful block challenge (blocker doesn't have card)."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 3
        player.cards = Counter({CARD.ASSASSIN: 2})

        target = player.next_alive
        target.cards = Counter({CARD.DUKE: 2})  # Doesn't have Contessa!
        initial_target_cards = target.cards.total()

        assassinate_idx = self.env.act_to_idx[(ACTION.ASSASSINATE, 1)]
        self.env.step(assassinate_idx)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        block_idx = self.env.act_to_idx[(ACTION.BLOCK_ASSASSINATE, 0)]
        self.env.step(block_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")
        self.assertEqual(self.env.agent_selection, target.id)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)

        challenge_call_idx = self.env.act_to_idx[(ACTION.CHALLENGE_CALL, 0)]
        self.env.step(challenge_call_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallengeResolve")
        self.assertEqual(self.env.turn.player.id, target.id)

        lose_duke_idx = self.env.act_to_idx[(ACTION.LOSE_DUKE, 0)]
        self.env.step(lose_duke_idx)

        self.assertEqual(target.cards.total(), initial_target_cards - 1)

        self.assertEqual(type(self.env.turn).__name__, "ActionResolve")

        self.env.step(lose_duke_idx)

        self.assertEqual(target.cards.total(), initial_target_cards - 2)

        self.assertEqual(type(self.env.turn).__name__, "EndTurn")
        self.assertFalse(target.alive)

        self.env.step(None)

        self.assertNotIn(target.id, self.env.agents)

    def test_failed_block_challenge(self):
        """Test failed block challenge (blocker DOES have card)."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 3
        player.cards = Counter({CARD.ASSASSIN: 2})

        target = player.next_alive
        target.cards = Counter({CARD.CONTESSA: 2})
        initial_target_cards = target.cards.total()

        assassinate_idx = self.env.act_to_idx[(ACTION.ASSASSINATE, 1)]
        self.env.step(assassinate_idx)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        block_idx = self.env.act_to_idx[(ACTION.BLOCK_ASSASSINATE, 0)]
        self.env.step(block_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallenge")
        self.assertEqual(self.env.agent_selection, target.id)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)

        third_player = target.next_alive
        challenge_call_idx = self.env.act_to_idx[(ACTION.CHALLENGE_CALL, 0)]
        self.env.step(challenge_call_idx)

        self.assertEqual(type(self.env.turn).__name__, "BlockChallengeResolve")
        self.assertEqual(self.env.turn.player.id, third_player.id)
        initial_third_cards = third_player.cards.total()

        third_card = next(iter(third_player.cards.elements()))
        lose_card_action = getattr(ACTION, f"LOSE_{third_card.name}")
        lose_idx = self.env.act_to_idx[(lose_card_action, 0)]
        self.env.step(lose_idx)

        self.assertEqual(third_player.cards.total(), initial_third_cards - 1)

        self.assertEqual(target.cards.total(), initial_target_cards)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestExchange(unittest.TestCase):
    """Test Exchange action mechanics."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_exchange_no_challenge(self):
        """Test Exchange action when not challenged."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.AMBASSADOR: 2})

        exchange_idx = self.env.act_to_idx[(ACTION.EXCHANGE, 0)]

        self.env.step(exchange_idx)

        self.assertEqual(type(self.env.turn).__name__, "Challenge")

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        self.assertEqual(type(self.env.turn).__name__, "ExchangeResolve")

        self.assertEqual(player.cards.total(), 4)

        lose_amb_idx = self.env.act_to_idx[(ACTION.LOSE_AMBASSADOR, 0)]
        self.env.step(lose_amb_idx)

        self.assertEqual(type(self.env.turn).__name__, "ExchangeTwoResolve")
        self.assertEqual(player.cards.total(), 3)

        self.env.step(lose_amb_idx)

        self.assertEqual(player.cards.total(), 2)

        self.assertEqual(type(self.env.turn).__name__, "Start")
        self.assertNotEqual(self.env.agent_selection, agent)


class TestGameEnd(unittest.TestCase):
    """Test game end conditions."""

    def setUp(self):
        """Create a 2-player environment for testing."""
        self.env = raw_env(num_players=2, num_players_alive=2, render_mode=None)
        self.env.reset(seed=42)

    def test_game_ends_when_one_player_left(self):
        """Test game ends when only one player has cards."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 7
        player.cards = Counter({CARD.DUKE: 1})

        target = player.next_alive
        target.cards = Counter({CARD.AMBASSADOR: 1})

        coup_idx = self.env.act_to_idx[(ACTION.COUP, 1)]
        self.env.step(coup_idx)

        self.assertEqual(type(self.env.turn).__name__, "ActionResolve")

        lose_amb_idx = self.env.act_to_idx[(ACTION.LOSE_AMBASSADOR, 0)]
        self.env.step(lose_amb_idx)

        self.assertFalse(target.alive)

        self.assertEqual(type(self.env.turn).__name__, "EndTurn")

        self.env.step(None)

        self.assertEqual(len(self.env.agents), 1)
        self.assertEqual(self.env.agents[0], player.id)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special situations."""

    def setUp(self):
        """Create a standard 3-player environment for testing."""
        self.env = raw_env(num_players=3, num_players_alive=3, render_mode=None)
        self.env.reset(seed=42)

    def test_steal_from_target_with_no_coins(self):
        """Test that stealing from a player with 0 coins is not allowed."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.CAPTAIN: 2})

        target = player.next_alive
        target.coins = 0

        steal_idx = self.env.act_to_idx[(ACTION.STEAL, 1)]
        action_mask = self.env.get_action_mask()
        self.assertFalse(action_mask[steal_idx])

    def test_steal_from_target_with_one_coin(self):
        """Test stealing from a player with only 1 coin."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.cards = Counter({CARD.CAPTAIN: 2})
        initial_coins = player.coins

        target = player.next_alive
        target.coins = 1

        steal_idx = self.env.act_to_idx[(ACTION.STEAL, 1)]
        self.env.step(steal_idx)

        challenge_pass_idx = self.env.act_to_idx[(ACTION.CHALLENGE_PASS, 0)]
        self.env.step(challenge_pass_idx)
        self.env.step(challenge_pass_idx)

        block_pass_idx = self.env.act_to_idx[(ACTION.BLOCK_PASS, 0)]
        self.env.step(block_pass_idx)

        self.assertEqual(player.coins, initial_coins + 1)
        self.assertEqual(target.coins, 0)

    def test_must_coup_with_10_coins(self):
        """Test that player with 10+ coins must Coup."""
        agent = self.env.agent_selection
        player = self.env.find_player(agent)
        player.coins = 10

        action_mask = self.env.get_action_mask()

        income_idx = self.env.act_to_idx[(ACTION.INCOME, 0)]
        self.assertFalse(action_mask[income_idx])

        coup_idx = self.env.act_to_idx[(ACTION.COUP, 1)]
        self.assertTrue(action_mask[coup_idx])


if __name__ == "__main__":
    unittest.main()
