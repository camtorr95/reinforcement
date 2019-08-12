import numpy as np

R = np.array([-1, -1, 1, 1, 1])
A = np.array([[-1, 0], [0, 1], [0, -1], [1, 0], [1, 0]])
Ai = np.array([0, 2, 3, 1, 1])
actions = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]])
epsilon = 0.5


def one_predicate(a):
    return np.isclose(A, a).all(axis=1).astype(int)


def estimated_value_Q_t(a):
    _one_predicate = one_predicate(a)
    sum_rewards = np.dot(R, _one_predicate)
    number_of_times = np.sum(_one_predicate)
    return 0 if number_of_times == 0 else sum_rewards / number_of_times


def greedy_action_selection_A_t():
    _Q_t = np.apply_along_axis(estimated_value_Q_t, 1, A)
    return actions[Ai[np.argmax(_Q_t)]]


def e_greedy_action_selection_A_t():
    _Q_t = np.apply_along_axis(estimated_value_Q_t, 1, A)
    random, greedy = np.random.choice(actions.shape[0]), Ai[np.argmax(_Q_t)]
    options, distribution = np.array([random, greedy]), np.array([epsilon, 1 - epsilon])
    return actions[np.random.choice(options, p=distribution)]


def main():
    print(e_greedy_action_selection_A_t())
    print('__main__')


if __name__ == '__main__':
    main()
