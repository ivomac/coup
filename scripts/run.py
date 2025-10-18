from coup_env.env.env import raw_env

env = raw_env()

for _ in range(100):
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            break

        action = env.action_space(agent).sample(info["action_mask"])
        env.step(action)

env.close()
