import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random as rnd


def generate_episode(states, termination_states, actions, grid_size, reward_size):
    init_state = rnd.choice(states[1:-1])
    episode = []
    while True:
        if list(init_state) in termination_states:
            return episode

        action = rnd.choice(actions)
        final_state = np.array(init_state) + np.array(action)

        if -1 in list(final_state) or grid_size in list(final_state):
            final_state = init_state

        episode.append([list(init_state), action, reward_size, list(final_state)])
        init_state = final_state


def monte_carlo_method(gamma, reward_size, grid_size, termination_states, actions, num_iterations):
    v = np.zeros((grid_size, grid_size))
    returns = {(i, j): list() for i in range(grid_size) for j in range(grid_size)}
    deltas = {(i, j): list() for i in range(grid_size) for j in range(grid_size)}
    states = [[i, j] for i in range(grid_size) for j in range(grid_size)]

    for it in tqdm(range(num_iterations)):
        episode = generate_episode(states, termination_states, actions, grid_size, reward_size)
        g = 0
        for i, step in enumerate(episode[::1]):
            g = gamma * g + step[2]
            if step[0] not in [x[0] for x in episode[::-1][len(episode) - i:]]:
                idx = (step[0][0], step[0][1])
                returns[idx].append(g)
                new_value = np.average(returns[idx])
                deltas[idx[0], idx[1]].append(np.abs(v[idx[0], idx[1]] - new_value))
                v[idx[0], idx[1]] = new_value

    # print(v)
    plt.figure(figsize=(20, 10))
    all_series = [list(x)[:50] for x in deltas.values()]
    for series in all_series:
        plt.plot(series)
    plt.show()


def main():
    _gamma = 0.6
    _reward_size = -1
    _grid_size = 4
    _termination_states = [[0, 0], [_grid_size - 1, _grid_size - 1]]
    _actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    _num_iterations = 10000
    monte_carlo_method(_gamma, _reward_size, _grid_size, _termination_states, _actions, _num_iterations)


if __name__ == '__main__':
    main()
