import unittest

from pettingzoo.test import api_test, seed_test

from pettingzoo_coup import coup_v0
from tests.common_params import PARAM_SETS


class TestPettingZooAPI(unittest.TestCase):
    def test_api_compliance(self):
        """Test that the environment complies with PettingZoo API standards."""
        for params in PARAM_SETS:
            with self.subTest(params=params):
                api_test(coup_v0.env(**params), num_cycles=1_000)

    def test_seed_consistency(self):
        """Test that the environment produces consistent results with the same seed."""
        seed_test(coup_v0.env, num_cycles=1_000)


if __name__ == "__main__":
    unittest.main()
