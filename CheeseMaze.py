import numpy as np

A, B, C, D, E, F, G, E2 = 1, 2, 3, 4, 5, 6, 7, 8
observations = (A, B, C, D, E, F, G)

UP, DOWN, LEFT, RIGHT = 1, 2, 3, 4
states = (UP, DOWN, LEFT, RIGHT)


class CheeseMaze:
    def __init__(self):
        self.maze = createMaze((5, 7))
        self.observations = observations
        self.states = states
        self.reward = 0
        self.reward_history = []
        self.init_position = (3, 1)
        self.cheese_position = (3, 3)
        self.position = self.init_position
        self.past_positions = []
        print(f"maze:\n{self.maze}")

    def getCurrentState(self):
        return self.maze, self.reward, self.position

    def getActionStates(self):
        return states

    def getObservationSpace(self):
        y, x = self.position
        return self.maze[y][x]

    def doAction(self, action: int):
        if action == UP:
            return self.__goUp()
        elif action == DOWN:
            return self.__goDown()
        elif action == LEFT:
            return self.__goLeft()
        elif action == RIGHT:
            return self.__goRight()
        else:
            raise ValueError(f"Action {action} is undefined")

    def __getReward(self, position):
        y, x = position
        if self.maze[y][x] == 0:
            return -10
        elif self.maze[y][x] == G:
            return 10
        else:
            return -1

    def __calculate_move(self):
        y, x = self.position
        if x == 0:  # hits wall
            self.position = self.past_positions[-1]
            self.reward -= 10
            self.reward_history.append(-10)
        else:
            self.reward_history.append(self.__getReward(self.position))
            self.reward += self.reward_history[-1]
        if self.maze[y][x] == G:
            self.position = self.init_position
        return self.reward_history[-1]

    def __goUp(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y - 1, x)
        reward = self.__calculate_move()
        return reward

    def __goDown(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y + 1, x)
        reward = self.__calculate_move()
        return reward

    def __goLeft(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y, x - 1)
        reward = self.__calculate_move()
        return reward

    def __goRight(self):
        self.past_positions.append(self.position)
        y, x = self.position
        self.position = (y, x + 1)
        reward = self.__calculate_move()
        return reward

    def getRewardTotal(self):
        return self.reward

    def getRewardLatest(self):
        return self.reward_history[-1]

    def mapObservationSpace(self, observation_space_old, observation_space_new):
        for i, (o1, o2) in enumerate(observation_space_new):
            if observation_space_old[i] == observation_space_new[i]:
                continue
            else:
                print(self.project_diff_to_state(o1, o2))

    def project_diff_to_state(self, o1, o2):
        point1 = None
        point2 = None
        figure = np.zeros(self.maze.shape)
        while not point1:
            y, x = np.randint(5), np.randint(7)
            if self.maze[y, x] == o1:
                point1 = y, x

        while not point2:
            y, x = np.randint(5), np.randint(7)
            if self.maze[y, x] == o2:
                point2 = y, x

        figure[point1] = 1
        figure[point2] = 2

        return figure


def createMaze(shape):
    maze = np.zeros(shape)  # outer 0 is all walls
    maze[1][1] = A
    maze[2][1] = E
    maze[3][1] = F
    maze[1][2] = B
    maze[1][3] = C
    maze[1][4] = B
    maze[1][5] = D
    maze[2][3] = E
    maze[3][3] = G
    maze[2][5] = E
    maze[3][5] = F
    return maze
