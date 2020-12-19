import numpy as np
import math
from deepgraphlib import Node

from CheeseMaze import CheeseMaze
from MountainCar import MountainCar

envBook = {"Cheesemaze": CheeseMaze, "MountainCar": MountainCar}


class Agent:
    def __init__(self, env):
        if env not in envBook:
            raise Exception(f"Environment {env} not found")
        self.env = envBook[env]()

    def doAction(self, action: int):
        return self.env.doAction(action)

    def getActionStates(self):
        return self.env.getActionStates()

    def getObservationSpace(self):
        return self.env.getObservationSpace()

    def getCurrentState(self):
        return self.env.getCurrentState()


class VDRAgent(Agent):
    def __init__(self, env):
        Agent.__init__(self, env)
        self.action_init = 0
        self.action = self.action_init
        self.N = 10
        self.optim = True

    def viterbi(self, O, S, pi, transition, em):
        t = np.zeros((len(S), len(O)))
        for s in range(len(S)):
            t[s, 0] = pi[s] * em[s, O[0]]
        for o in range(len(O)):
            for s in range(len(S)):
                k = np.argmax(k in t[k, o - 1] * transition[k, s] * em[s, o])
                t[s, o] = t[k, o - 1] * transition[k, s] * em[s, o]
        best_path = []
        for o in range(0, len(O), -1):
            best_path.insert(0, S[np.argmax(k in t[k, o])])
        return best_path

    def VDR(self, O, episodes):
        D = []
        for _ in episodes:
            pi, v = self.OPTE(D)
            o_new, D_prime, v_prime = None, None, None
            for o in O:
                o_new = o.getAugmentation()
                D_prime = o.relabel(o_new[0], o_new[1])
                pi, v = self.OPTE(D_prime)
                if pi * (v_prime - v) > 0.25:
                    o = o.update_with_split(o_new)

    def OPTE(self, D):
        R = [[]] * self.N
        for i in range(self.N):
            node = Node(D[0])
            o = node.obs()
            done = False
            while not done:
                node, r, done = self.next(node, o)
                R[i].append(r)
                o = node.obs()
        return sum([sum(r) for r in R]) / self.N

    def next(self, node, o):
        C = 1
        counts = node.getTransitionCounts(self.action)
        counts.append(C)
        model = counts.MLE()
        nextNode = model(self.action)
        r = None
        if not nextNode:
            o = node.obs()
            done = True
            if self.optim:
                r = node.getQ(o, self.action) + r * math.sqrt(math.log2(o) / node.obs())
            else:
                r = node.getQ(o, self.action)
        else:
            r, done = nextNode.getInfo()
        return nextNode, r, done

    """
    Some of the following code was guided by https://www.kaggle.com/charel/learn-by-example-expectation-maximization
    """

    def genPdf(self, sigma, mu):
        def pdf(self):
            u = (sigma - self.mu) / abs(sigma)
            y = (1 / (math.sqrt(2 * 3.14) * abs(sigma))) * math.exp(-u * u / 2)
            return y

        return pdf

    def em_init(self, data, min, max, sigma=1, mix=0.5):
        self.data = data
        self.one = self.genPdf(sigma, min)
        self.two = self.genPdf(sigma, max)

        self.mix = mix

    def Estep(self):
        logres = 0
        for s_loc in self.data:
            wp1 = self.one(s_loc) * self.mix
            wp2 = self.two(s_loc) * (1.0 - self.mix)
            den = wp1 + wp2
            wp1 /= den
            wp2 /= den
            logres += math.log2(den)
            yield (wp1, wp2)

    def Mstep(self, weights):
        (left, rigt) = zip(*weights)
        one_den = sum(left)
        two_den = sum(rigt)

        self.one.mu = sum(w * d for (w, d) in zip(left, self.data)) / one_den
        self.two.mu = sum(w * d for (w, d) in zip(rigt, self.data)) / two_den

        self.one.sigma = math.sqrt(
            sum(w * ((d - self.one.mu) ** 2) for (w, d) in zip(left, self.data))
            / one_den
        )
        self.two.sigma = math.sqrt(
            sum(w * ((d - self.two.mu) ** 2) for (w, d) in zip(rigt, self.data))
            / two_den
        )
        self.mix = one_den / len(self.data)

    def iterate(self, N=1, verbose=False):
        for i in range(1, N + 1):
            self.Mstep(self.Estep())
            if verbose:
                print("{0:2} {1}".format(i, self))
        self.Estep()
