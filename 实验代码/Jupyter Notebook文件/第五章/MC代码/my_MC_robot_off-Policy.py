from collections import defaultdict
from myBookEnvCode.my_book_gridworld0406 import GridWorldEnv
import numpy as np

class Agent():
    def __init__(self):
        self.Q = {}  # {state:{action:q_value}} 初始化Q ★
        self.C = {}

    def create_random_policy(self,nA):
        """
        Creates a random policy function.

        Args:
            nA: Number of actions in the environment.

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities
        """

        def policy_fn(action_list,observation):
            A = np.zeros(nA, dtype=float)
            valid_nA=len(action_list[observation])
            for action in action_list[observation]:
                A[action]=1./valid_nA
            return A

        return policy_fn

    def create_greedy_policy(self,nA):
        """
        Creates a greedy policy based on Q values.

        Args:
            Q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(state):
            A = np.zeros(nA, dtype=float)
            best_action = max(self.Q[state], key=self.Q[state].get)  # 获取最大value值对应的key，即得到最大动作值函数对应的动作
            A[best_action] = 1.0
            return A

        return policy_fn

    def mc_control_importance_sampling(self, env, gamma, max_episode_num):
        behavior_policy=self.create_random_policy(env.action_space.n)
        target_policy=self.create_greedy_policy(env.action_space.n)
        num_episode = 0
        while num_episode <= max_episode_num:
            episode=[]
            state= env.reset()

            for t in range(100):
                if not self.__isStateInQ(state):
                    self.initValue(state, env.valid_actions_for_states, randomized=False)
                if not self.__isStateInC(state):
                    self.initCValue(state, env.valid_actions_for_states)
                probs=behavior_policy(env.valid_actions_for_states,state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state,reward,done,_=env.step(action)
                episode.append((state,action,reward))
                if done:
                    # print(t)
                    break
                state=next_state

            num_episode+=1
            # Sum of discounted returns
            G = 0.0
            # The importance sampling ratio (the weights of the returns)
            W = 1.0
            # For each step in the episode, backwards
            for t in range(len(episode))[::-1]:
                state, action, reward = episode[t]
                # Update the total reward since step t
                G = gamma * G + reward
                # Update weighted importance sampling formula denominator
                pre_c=self.__getCValue(state,action)
                self.__setCValue(state,action,pre_c+W)
                # Update the action-value function using the incremental update formula (5.7)
                # This also improves our target policy which holds a reference to Q
                now_c=self.__getCValue(state,action)
                pre_Q=self.__getQValue(state,action)
                self.__setQValue(state,action,pre_Q+(W / now_c) * (G - pre_Q))
                # If the action taken by the behavior policy is not the action
                # taken by the target policy the probability will be 0 and we can break
                if action != np.argmax(target_policy(state)):
                    break
                W = W * 1. / behavior_policy(env.valid_actions_for_states,state)[action]

            if num_episode in print_episodes:
                print("episode:{}".format(num_episode))
                for s in print_states:
                    if s in self.Q.keys():
                        print("{}_Q:".format(s), end="")
                        Q_s = []
                        for a in self.Q[s].keys():
                            Q_s.append(round(self.Q[s][a], 2))
                        print(Q_s)
        return self.Q

    # return a possible action list for a given state
    # def possibleActionsForstate(self, state):
    #  actions = []
    #  # add your code here
    #  return actions

    # if a state exists in Q dictionary
    def __isStateInQ(self, state):
        # 判断空值。有值则返回值，无值则返回None - None is not None = False
        return self.Q.get(state) is not None  # 因为是实例属性，所以要用self.进行引用

    # if a state exists in C dictionary
    def __isStateInC(self, state):
        # 判断空值。有值则返回值，无值则返回None - None is not None = False
        return self.C.get(state) is not None  # 因为是实例属性，所以要用self.进行引用

    def initValue(self, s, valid_actions_list,randomized=False):  # 初始化Q
        # Q[s]为空值时进入判断
        if not self.__isStateInQ(s):
            self.Q[s] = {}  # 初始化Q
            for a in valid_actions_list[s]:  # 遍历所有action_name
                self.Q[s][a] = np.random.random() / 10 if randomized is True else 0.0  # 初始化Q(s,a)；随机一个动作值函数。只有结束状态的Q(s,a) = 0

    def initCValue(self, s, valid_actions_list):  # 初始化C
        if not self.__isStateInC(s):
            self.C[s] = {}  # 初始化C
            for a in valid_actions_list[s]:  # 遍历所有action_name
                self.C[s][a] = 0.0  # 初始化C(s,a)值为0.0

    """Q的获取与设置方法"""

    def __getQValue(self, s, a):  # ①
        return self.Q[s][a]  # argmax(q)

    def __setQValue(self, s, a, new_q):  # ②
        self.Q[s][a] = new_q

    """C的获取与设置方法"""

    def __getCValue(self, s, a):
        return self.C[s][a]

    def __setCValue(self, s, a, new_c):
        self.C[s][a] = new_c

np.random.seed(1)
agent=Agent()
env=GridWorldEnv()
print_states=[5,10,18,20,24]
print_episodes=[1,7500,12500,19999,20000]
Q=agent.mc_control_importance_sampling(env=env,gamma=0.8,max_episode_num=20000)