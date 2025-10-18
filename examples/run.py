from pettingzoo_coup import coup_v0

env = coup_v0.env()

env.reset()

print(env.render())

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    # info also contains past turn observations in info["observation_history"]

    action: int | None

    if termination or truncation:
        action = None
    else:
        action_mask = info["action_mask"]

        # here we take random actions sampled from the masked action space
        # replace with policy
        action = env.action_space(agent).sample(action_mask)

    env.step(action)

    print(env.render())

env.close()
