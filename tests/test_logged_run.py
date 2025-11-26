import logging
import unittest
from pathlib import Path

from pettingzoo_coup import coup_v0
from tests.common_params import PARAM_SETS

SEEDS = list(range(10))
LOG_DIR = Path("logs")


def setup_logging(params, seed):
    """Set up logging to a file specific to params and seed."""
    params_str = f"players_{params['num_players']}_alive_{params['num_players_alive']}_dead_draw_{int(params['dead_draw'])}"
    if "deck" in params:
        params_str += "_" + "_".join(f"{k[:3]}_{v}" for k, v in params["deck"].items())

    filename = f"seed_{seed}_{params_str}.log"
    filepath = LOG_DIR / filename

    LOG_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        if hasattr(handler, "close"):
            handler.close()

    file_handler = logging.FileHandler(filepath, mode="w")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return file_handler


def run(seed, **kwargs):
    setup_logging(kwargs, seed)

    env = coup_v0.env(**kwargs)

    env.reset(seed=seed)

    logging.debug(env.render())

    for agent in env.agent_iter():
        _, _, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            action_mask = info.get("action_mask", None)
            action = env.action_space(agent).sample(action_mask)

        env.step(action)

        logging.debug(env.render())

    env.close()


class TestLoggedRun(unittest.TestCase):
    def test_logged_run(self):
        """Run the game and capture logs to separate files."""
        for params in PARAM_SETS:
            for seed in SEEDS:
                with self.subTest(params=params, seed=seed):
                    run(seed=seed, **params)


if __name__ == "__main__":
    unittest.main()
