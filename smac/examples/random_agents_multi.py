from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from smac.env import StarCraft2Env


def main():
    env = StarCraft2Env(map_name="3m_multi_close", heuristic_enemy=False,
                        window_size_x=1024, window_size_y=600, debug=True, obs_last_action=True)

    env_info = env.get_env_info()

    n_agents_p1 = env_info["n_agents"]
    n_agents_p2 = env_info["n_enemies"]

    n_episodes = 10

    for e in range(n_episodes):
        env.reset()

        terminated = False
        episode_reward = np.zeros(2)
        cpt = 0
        while not terminated:
            cpt += 1
            obs = env.get_obs()
            state = env.get_state()

            actions = []
            for agent_id in range(n_agents_p1):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            for agent_id in range(n_agents_p1, n_agents_p1 + n_agents_p2):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total time duration          {}".format(cpt))
        print("Total reward in episode {} = {}".format(e, episode_reward))
    env.close()


if __name__ == "__main__":
    main()
