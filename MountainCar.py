import numpy as np


class MountainCar:
    def __init__(self):
        self.stateSpace = createstateSpace((8, 8))
        self.observations = [0, 1]
        self.velocity = 0
        self.timeSteps = 0
        self.position = 0.5
        self.reward = 0
        self.finish_position = 1
        self.past_positions = []
        self.gravity = 9.8
        self.delta = 0.01
        print(f"stateSpace:\n{self.stateSpace}")

    def step(self, action):
        if not -1 <= action <= 1:
            raise ValueError(f"Action {action} is undefined")
        done = False
        reward = -0.01
        action_list = [-0.2, 0, +0.2]
        action_move = action_list[action + 1]
        velocity_post = (
            self.velocity
            + (-3 * (self.gravity * np.cos(self.position) + action_move / 3))
            * self.delta
        )
        position_post = self.position + (velocity_post * self.delta)
        if position_post < -1.2:
            position_post = -1.2
            velocity_post = 0
        self.position = position_post
        self.velocity = velocity_post
        self.past_positions.append(position_post)
        return [position_post, velocity_post]

    def getCurrentState(self):
        return self.stateSpace, self.velocity, self.position

    def getActionStates(self):
        return [-1, 0, 1]

    def getObservationSpace(self):
        y, x = self.position
        return self.stateSpace[y][x]

    def doAction(self, action: int):
        if action == 0:
            None
        elif action == -1:
            return self.__goLeft()
        elif action == 1:
            return self.__goRight()
        else:
            raise ValueError(f"Action {action} is undefined")

    def __getReward(self, position):
        y, x = position
        if self.stateSpace[y][x] == 0:
            return -10
        elif self.stateSpace[y][x] == G:
            return 10
        else:
            return -1

    def __goLeft(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y, x - 1)
        reward = -1
        return reward

    def __goRight(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y, x + 1)
        reward = -1
        return reward

    def getRewardTotal(self):
        return self.reward

    def mapObservationSpace(self, observation_space_old, observation_space_new):
        for i, (o1, o2) in enumerate(observation_space_new):
            if observation_space_old[i] == observation_space_new[i]:
                continue
            else:
                print(self.project_diff_to_state(o1, o2))

    def project_diff_to_state(self, o1, o2):
        point1 = None
        point2 = None
        figure = np.zeros(self.stateSpace.shape)
        while not point1:
            y, x = np.randint(7), np.randint(7)
            if self.stateSpace[y, x] == o1:
                point1 = y, x

        while not point2:
            y, x = np.randint(7), np.randint(7)
            if self.stateSpace[y, x] == o2:
                point2 = y, x

        figure[point1] = 1
        figure[point2] = 2

        return figure


def createstateSpace(shape):
    stateSpace = np.zeros(shape)
    stateSpace[:, :4] = 0
    stateSpace[:, 4:] = 1
    return stateSpace
