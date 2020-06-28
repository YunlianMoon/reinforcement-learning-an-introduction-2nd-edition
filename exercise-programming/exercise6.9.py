import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# add four additional actions and stay action

matplotlib.use('Agg')

# world height
WORLD_HEIGHT = 7

# world width
WORLD_WIDTH = 10

# wind strength for each column
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_UPLEFT = 4
ACTION_UPRIGHT = 5
ACTION_DOWNLEFT = 6
ACTION_DOWNRIGHT = 7
ACTION_STAY = 8

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# reward for each step
REWARD = -1.0

START = [3, 0]
GOAL = [3, 7]


def step(state, action):
    i, j = state
    if action == ACTION_UP:
        return [max(i - 1 - WIND[j], 0), j]
    elif action == ACTION_DOWN:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), j]
    elif action == ACTION_LEFT:
        return [max(i - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        return [max(i - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_UPLEFT:
        return [max(i - 1 - WIND[j], 0), max(j - 1, 0)]
    elif action == ACTION_UPRIGHT:
        return [max(i - 1 - WIND[j], 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWNLEFT:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), max(j - 1, 0)]
    elif action == ACTION_DOWNRIGHT:
        return [max(min(i + 1 - WIND[j], WORLD_HEIGHT - 1), 0), min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_STAY:
        return [i, j]
    else:
        assert False


# play for an episode
def episode(q_value, actions):
    ACTIONS = actions

    # track the total time steps in this episode
    time = 0

    # initialize state
    state = START

    # choose an action based on epsilon-greedy algorithm
    if np.random.binomial(1, EPSILON) == 1:
        action = np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        action = np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

    # keep going until get to the goal state
    while state != GOAL:
        next_state = step(state, action)
        if np.random.binomial(1, EPSILON) == 1:
            next_action = np.random.choice(ACTIONS)
        else:
            values_ = q_value[next_state[0], next_state[1], :]
            next_action = np.random.choice(
                [action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        # Sarsa update
        q_value[state[0], state[1], action] += \
            ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
                     q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        time += 1
    return time


def figure_exercise_6_9(use_stay_action=False):
    if use_stay_action:
        actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UPLEFT, ACTION_UPRIGHT, ACTION_DOWNLEFT,
                   ACTION_DOWNRIGHT, ACTION_STAY]
    else:
        actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UPLEFT, ACTION_UPRIGHT, ACTION_DOWNLEFT,
                   ACTION_DOWNRIGHT]

    # num_action = np.shape(actions)[0]
    num_action = len(actions)

    q_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, num_action))
    episode_limit = 500

    steps = []
    ep = 0
    while ep < episode_limit:
        steps.append(episode(q_value, actions))
        # time = episode(q_value)
        # episodes.extend([ep] * time)
        ep += 1

    steps = np.add.accumulate(steps)

    plt.plot(steps, np.arange(1, len(steps) + 1))
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')

    if use_stay_action:
        plt.savefig('../images/exercise_6_9_9.png')
    else:
        plt.savefig('../images/exercise_6_9_8.png')
    plt.close()

    # display the optimal policy
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
            elif bestAction == ACTION_UPLEFT:
                optimal_policy[-1].append('UL')
            elif bestAction == ACTION_UPRIGHT:
                optimal_policy[-1].append('UR')
            elif bestAction == ACTION_DOWNLEFT:
                optimal_policy[-1].append('DL')
            elif bestAction == ACTION_DOWNRIGHT:
                optimal_policy[-1].append('DR')
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND]))


if __name__ == '__main__':
    # figure_exercise_6_9(use_stay_action=False)
    figure_exercise_6_9(use_stay_action=True)
