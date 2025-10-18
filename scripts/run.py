import logging

from coup_pettingzoo.env.env import raw_env

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

env = raw_env()

env.reset()

env.render()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break

    action = env.action_space(agent).sample(info["action_mask"])
    env.step(action)

    env.render()

env.close()
