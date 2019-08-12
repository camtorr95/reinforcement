import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def action_reward_function(terminationstates, reward_size, initial_position, action):
    if initial_position in terminationstates:
        return initial_position, 0

    reward = reward_size
    final_position = np.array(initial_position) + np.array(action)

    if -1 in final_position or 4 in final_position:
        final_position = initial_position

    return final_position, reward


def iterative_policy_evaluation(gamma, reward_size, grid_size, termination_states, actions, num_iterations):
    states = [[i, j] for i in range(grid_size) for j in range(grid_size)]
    value_map = np.zeros((grid_size, grid_size))
    deltas = []

    for it in tqdm(range(num_iterations)):
        _copyvalue_map = np.copy(value_map)
        _delta_state = []

        for _state in states:
            _weighted_rewards = 0

            for _action in actions:
                _final_position, _reward = action_reward_function(termination_states, reward_size, _state, _action)
                _factor = _reward + (gamma * value_map[_final_position[0], _final_position[1]])
                _weighted_rewards += (1 / len(actions)) * _factor

            _delta_state.append(np.abs(_copyvalue_map[_state[0], _state[1]] - _weighted_rewards))
            _copyvalue_map[_state[0], _state[1]] = _weighted_rewards

        deltas.append(_delta_state)
        value_map = _copyvalue_map
        '''
        if it in [0, 1, 2, 9, 99, num_iterations - 1]:
            print("Iteration {}".format(it + 1))
            print(value_map)
            print("")
        '''
    plt.figure(figsize=(20, 10))
    plt.plot(deltas)
    plt.show()


def main():
    _gamma = 1
    _reward_size = -1
    _grid_size = 4
    _termination_states = [[0, 0], [_grid_size - 1, _grid_size - 1]]
    _actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
    _num_iterations = 1000
    iterative_policy_evaluation(_gamma, _reward_size, _grid_size, _termination_states, _actions, _num_iterations)


if __name__ == '__main__':
    main()
