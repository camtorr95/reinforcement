import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random as rnd


def generate_initial_state(states):
    init_state = rnd.choice(states[1:-1])
    return init_state


def generate_next_action(actions):
    return rnd.choice(actions)


def take_action(termination_states, state, action, reward_size, grid_size):
    if list(state) in termination_states:
        return 0, None

    final_state = np.array(state)+np.array(action)
    # if robot crosses wall
    if -1 in list(final_state) or grid_size in list(final_state):
        final_state = state

    return reward_size, list(final_state)


def temporal_difference_learning(gamma, reward_size, grid_size, alpha, termination_states, actions, num_iterations):
    v = np.zeros((grid_size, grid_size))
    returns = {(i, j): list() for i in range(grid_size) for j in range(grid_size)}
    deltas = {(i, j): list() for i in range(grid_size) for j in range(grid_size)}
    states = [[i, j] for i in range(grid_size) for j in range(grid_size)]

    for it in tqdm(range(num_iterations)):
        state = generate_initial_state(states)
        while True:
            action = generate_next_action(actions)
            reward, final_state = take_action(termination_states, state, action, reward_size, grid_size)

            # we reached the end
            if final_state is None:
                break

            # modify Value function
            before = v[state[0], state[1]]
            factor = reward + gamma * v[final_state[0], final_state[1]] - v[state[0], state[1]]
            v[state[0], state[1]] += alpha * factor
            deltas[state[0], state[1]].append(float(np.abs(before - v[state[0], state[1]])))

            state = final_state

    print(v)

    plt.figure(figsize=(20, 10))
    all_series = [list(x)[:50] for x in deltas.values()]
    for series in all_series:
        plt.plot(series)
    plt.show()


def main():
    _gamma = 0.1  # discounting rate
    _reward_size = -1
    _grid_size = 4
    _alpha = 0.1  # (0,1] // stepSize
    _termination_states = [[0, 0], [_grid_size - 1, _grid_size - 1]]
    _actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    _num_iterations = 10000
    temporal_difference_learning(_gamma, _reward_size, _grid_size, _alpha,
                                 _termination_states, _actions, _num_iterations)


if __name__ == '__main__':
    main()
