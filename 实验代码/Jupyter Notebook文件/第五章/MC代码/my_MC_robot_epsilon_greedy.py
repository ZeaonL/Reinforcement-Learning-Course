from collections import defaultdict
from myBookEnvCode.my_book_gridworld0406 import GridWorldEnv
import numpy as np

class Agent():
    def __init__(self):
        self.Q = {}  # {state:{action:q_value}} 初始化Q ★

    def create_epsilon_greedy_policy(self,nA):
        """
        Creates an epsilon-greedy policy based on Q values.

        Args:
            Q: A dictionary that maps from state -> action values

        Returns:
            A function that takes an observation as input and returns a vector
            of action probabilities.
        """

        def policy_fn(action_list,state,epsilon):
            A = np.zeros(nA, dtype=float)
            valid_nA = len(action_list[state])
            for action in action_list[state]:
                A[action] = epsilon / valid_nA
            best_action = max(self.Q[state], key=self.Q[state].get)  # 获取最大value值对应的key，即得到最大动作值函数对应的动作
            A[best_action] += 1.0-epsilon
            return A

        return policy_fn

    def mc_control_epsilon_greedy(self, env, gamma, max_episode_num):
        # flag = True
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)
        target_policy = self.create_epsilon_greedy_policy(env.action_space.n)
        num_episode = 0
        for state in range(env.observation_space.n):
            self.initValue(state, env.valid_actions_for_states, randomized=False)
        while num_episode < max_episode_num:
            episode=[]
            state= env.reset()
            while True:
                #env.render()
                probs = target_policy(env.valid_actions_for_states, state,epsilon_by_epsiode(num_episode))
                action = np.random.choice(np.arange(len(probs)), p=probs)
                next_state,reward,done,_=env.step(action)
                episode.append((state,action,reward))
                if done:
                    break
                state=next_state

            num_episode+=1
            # Find all (state, action) pairs we've visited in this episode
            # We convert each state to a tuple so that we can use it as a dict key
            sa_in_episode = set([(x[0], x[1]) for x in episode])
            for state, action in sa_in_episode:
                sa_pair = (state, action)
                # Find the first occurance of the (state, action) pair in the episode
                first_occurence_idx = next(i for i, x in enumerate(episode)
                                            if x[0] == state and x[1] == action)
                # Sum up all rewards since the first occurance
                G = sum([x[2] * (gamma ** i) for i, x in enumerate(episode[first_occurence_idx:])])
                # Calculate average return for this state over all sampled episodes
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                self.__setQValue(state,action,returns_sum[sa_pair] / returns_count[sa_pair])


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

    def initValue(self, s, valid_actions_list,randomized=False):  # 初始化Q和E
        # Q[s]为空值时进入判断
        if not self.__isStateInQ(s):
            self.Q[s] = {}  # 初始化Q
            for a in valid_actions_list[s]:  # 遍历所有action_name
                self.Q[s][a] = np.random().random() / 10 if randomized is True else 0.0  # 初始化Q(s,a)；随机一个动作值函数。只有结束状态的Q(s,a) = 0


    """Q的获取与设置方法"""

    def __getQValue(self, s, a):  # ①
        return self.Q[s][a]  # argmax(q)

    def __setQValue(self, s, a, new_q):  # ②
        self.Q[s][a] = new_q

np.random.seed(1)
epsilon_start = 0.5
epsilon_final = 0
epsilon_episodes = 20000
epsilon_by_epsiode = lambda episode_idx: epsilon_start - (epsilon_start - epsilon_final) * min(episode_idx,
                                                                                                   epsilon_episodes) / epsilon_episodes
agent=Agent()
env=GridWorldEnv()
print_states=[5,10,18,20,24]
print_episodes=[1,7500,12500,19999,20000]
Q=agent.mc_control_epsilon_greedy(env=env,gamma=0.8,max_episode_num=20000)
