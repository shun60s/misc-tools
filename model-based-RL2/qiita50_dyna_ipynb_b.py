# -*- coding: utf-8 -*-
"""

「モデルベース強化学習及びプランニング」
  簡単な４ｘ４マスの中を移動するテーブル型モデル(GridEnv)を実装して使っている。
  近似モデルとしてマルコフ決定過程(MDP)の学習を行う。
  学習したモデルから価値または方策を,Q学習または方策反復法で学習する(プランニング)。

　モデルとして、状態遷移関数と終了関数は確率を扱うので確率分布として、
　報酬関数は報酬の期待値でモデル化する。

これは、
https://colab.research.google.com/drive/1Le_lPzGrjBQ7cZK1vktg9xoRnDcxYzIi?usp=sharing
qiita50_dyna.ipynbのコピーを一部編集したものです。

https://qiita.com/pocokhc/items/bcebc4b7b2454028baf9
に説明があります。
"""

"""
check version
python 3.6.4 on win32
gym 0.15.3
numpy 1.19.5
matplotlib 3.3.1
"""



"""# 1.共通コード

## 1-1. import
"""
import gym
import numpy as np
from matplotlib import pyplot as plt

import collections
import random
import copy


#-------------------------------------------------------------------------------------------------
"""
## 1-2.GridEnvクラスの実装

"""

import enum
class Action(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridEnv(gym.Env):
    def __init__(self, move_prob=0.8):
        super().__init__()

        self.grid = [
            [0, 0, 0, 1],
            [0, 9, 0, -1],
            [0, 0, 0, 0],
        ]

        self.agent_state = None
        self.default_reward = -0.04

        # 遷移確率
        self.move_prob = {
            Action.UP.value: {
                Action.UP.value: move_prob,
                Action.DOWN.value: 0,
                Action.RIGHT.value: (1 - move_prob) / 2,
                Action.LEFT.value: (1 - move_prob) / 2,
            },
            Action.DOWN.value: {
                Action.UP.value: 0,
                Action.DOWN.value: move_prob,
                Action.RIGHT.value: (1 - move_prob) / 2,
                Action.LEFT.value: (1 - move_prob) / 2,
            },
            Action.RIGHT.value: {
                Action.UP.value: (1 - move_prob) / 2,
                Action.DOWN.value: (1 - move_prob) / 2,
                Action.RIGHT.value: move_prob,
                Action.LEFT.value: 0,
            },
            Action.LEFT.value: {
                Action.UP.value: (1 - move_prob) / 2,
                Action.DOWN.value: (1 - move_prob) / 2,
                Action.RIGHT.value: 0,
                Action.LEFT.value: move_prob,
            },
        }

        self.viewer = None

        # super: action_space
        self.action_space = gym.spaces.Discrete(4)
        # super: observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.H, self.W),
        )
        self.reset()

    def __enter__(self):
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
    def __del__(self):
        self.close()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @property
    def W(self):
        return len(self.grid[0])

    @property
    def H(self):
        return len(self.grid)

    @property
    def actions(self):
        return [
            Action.UP.value,
            Action.DOWN.value,
            Action.LEFT.value,
            Action.RIGHT.value,
        ]

    @property
    def states(self):
        states = []
        for y in range(self.H):
            for x in range(self.W):
                #if self.grid[y][x] != 9:
                    states.append((x, y))
        return states

    def reset(self):  # override
        self.agent_state = (0, self.H-1)
        self.step_count = 0
        return self.agent_state

    def step(self, action):  # override
        next_state, reward, done = self._transit(self.agent_state, action)
        if next_state is not None:
            self.agent_state = next_state

        self.step_count += 1
        return next_state, reward, done, {}

    def can_action_at(self, state):
        if self.grid[state[1]][state[0]] == 0:
            return True
        else:
            return False

    # 次の状態へ遷移する確率とその状態
    def transitions_at(self, state, action):
        transition_probs = {}
        for a in self.actions:
            prob = self.move_prob[action][a]
            tmp_state = self._move(state, a)

            if tmp_state not in transition_probs:
                transition_probs[tmp_state] = 0
            transition_probs[tmp_state] += prob
        return transition_probs

    def reward_func(self, state):
        reward = self.default_reward
        done = False

        attribute = self.grid[state[1]][state[0]]
        if attribute == 1:
            reward = 1
            done = True
        elif attribute == -1:
            reward = -1
            done = True

        return reward, done

    def _transit(self, state, action):

        # 移動できない
        if not self.can_action_at(state):
            return None, 0, True

        # 次の状態へ遷移する確率とその状態
        transition_probs = self.transitions_at(state, action)

        # ランダムに決定
        next_states = list(transition_probs.keys())
        probs = list(transition_probs.values())
        next_state = next_states[np.random.choice(len(next_states), p=probs)]

        # 報酬
        reward, done = self.reward_func(next_state)

        return next_state, reward, done


    def _move(self, state, action):
        next_state = list(state)

        if action == Action.UP.value:
            next_state[1] -= 1
        elif action == Action.DOWN.value:
            next_state[1] += 1
        elif action == Action.LEFT.value:
            next_state[0] -= 1
        elif action == Action.RIGHT.value:
            next_state[0] += 1
        else:
            raise ValueError()

        # check
        if not (0 <= next_state[0] < self.W):
            next_state = state
        if not (0 <= next_state[1] < self.H):
            next_state = state

        # 移動できない
        if self.grid[next_state[1]][next_state[0]] == 9:
            next_state = state

        return tuple(next_state)


    def render(self, mode='human'):  # override

        print("step: {}".format(self.step_count))
        for y in range(self.H):
            s = ""
            for x in range(self.W):
                n = self.grid[y][x]

                if self.agent_state == (x, y):  # player
                    s += "P"
                elif n == 9:  # 壁
                    s += "x"
                elif n == 0:  # 道
                    s += " "
                elif n == 1:  # goal
                    s += "G"
                elif n == -1:  # 穴
                    s += "X"
            print(s)
        print("")

        return

#-------------------------------------------------------------------------------------------------
"""# 2. 近似モデルクラス(A_MDP)"""

class A_MDP():
    def __init__(self, env):
        self.env = env  # print用

        self.env_states = env.states
        self.env_actions = env.actions

        #--- モデルの初期化
        self.trans = {}   # [state][action][next_state] = 訪れた回数
        self.reward = {}  # [state][action] = 得た報酬の合計
        self.done = {}    # [state][action] = 終了した回数
        self.count = {}   # [state][action] = 訪れた回数

        # 初期化
        for s in env.states:
            self.reward[s] = [0.0] * len(env.actions)
            self.done[s] = [0] * len(env.actions)
            self.count[s] = [0] * len(env.actions)
            self.trans[s] = {}
            for a in env.actions:
                self.trans[s][a] = {}
                for s2 in env.states:
                    self.trans[s][a][s2] = 0

        # サンプリング用に実際にあった履歴を保存
        self.state_action_history = []

    # 全状態を返す
    @property
    def states(self):
        return self.env_states

    # 全アクションを返す
    @property
    def actions(self):
        return self.env_actions


    # 学習
    def train(self, state, action, n_state, reward, done):
        # (状態,アクション)の履歴を保存
        self.state_action_history.append([state, action])

        # 各回数をカウント
        self.count[state][action] += 1
        self.trans[state][action][n_state] += 1
        self.done[state][action] += 1 if done else 0
        self.reward[state][action] += reward

        assert self.count[state][action] == sum(self.trans[state][action].values())

    # ランダムに履歴を返す
    def samples(self, num):
        return random.sample(self.state_action_history, num)

    # 次の状態を返す
    def sample_next_state(self, state, action):
        weights = list(self.trans[state][action].values())
        n_s_list = list(self.trans[state][action].keys())
        n_s = random.choices(n_s_list, weights=weights, k=1)[0]
        return n_s

    # 報酬を返す
    def get_reward(self, state, action):
        if self.count[state][action] == 0:
            return 0
        return self.reward[state][action] / self.count[state][action]

    # 終了状態を返す
    def sample_done(self, state, action):
        if self.count[state][action] == 0:
            return (random.random() < 0.5)
        return (random.random() < (self.done[state][action] / self.count[state][action]))

    # 次の状態の遷移確率を返す
    def get_next_state_probs(self, state, action):
        probs = {}
        for s2, s2_count in self.trans[state][action].items():
            if self.count[state][action] == 0:
                probs[s2] = 0
            else:
                probs[s2] = s2_count / self.count[state][action]
        return probs

    def print(self):
        r = copy.deepcopy(self.reward)
        for s in self.env.states:
            for a in self.env.actions:
                if self.count[s][a] == 0:
                    continue
                r[s][a] = r[s][a] / self.count[s][a]
        print("報酬関数 A_MDP")
        print("------------------------------------------------")
        for y in range(self.env.H):
            # 上
            s = ""
            for x in range(self.env.W):
                s += "   {:5.2f}   |".format(r[(x,y)][Action.UP.value])
            print(s)
            # 左右
            s = ""
            for x in range(self.env.W):
                s += "{:5.2f} {:5.2f}|".format(r[(x,y)][Action.LEFT.value], r[(x,y)][Action.RIGHT.value])
            print(s)
            # 下
            s = ""
            for x in range(self.env.W):
                s += "   {:5.2f}   |".format(r[(x,y)][Action.DOWN.value])
            print(s)
            print("------------------------------------------------")

        d = copy.deepcopy(self.done)
        for s in self.env.states:
            for a in self.env.actions:
                if self.count[s][a] == 0:
                    continue
                d[s][a] = 100 * d[s][a] / self.count[s][a]

        print("終了関数 A_MDP")
        print("-------------------------------------------------------")
        for y in range(self.env.H):
            # 上
            s = ""
            for x in range(self.env.W):
                s += "    {:5.1f}%   |".format(d[(x,y)][Action.UP.value])
            print(s)
            # 左右
            s = ""
            for x in range(self.env.W):
                s += "{:5.1f}% {:5.1f}%|".format(d[(x,y)][Action.LEFT.value], d[(x,y)][Action.RIGHT.value])
            print(s)
            # 下
            s = ""
            for x in range(self.env.W):
                s += "    {:5.1f}%   |".format(d[(x,y)][Action.DOWN.value])
            print(s)
            print("-------------------------------------------------------")

        print("遷移関数 A_MDP")
        for s in self.env.states:
            for a in self.env.actions:
                for s2 in self.env.states:
                    if self.trans[s][a][s2] == 0:
                        continue
                    print("{}:{} -> {}:{:.1f}%".format(
                        s, Action(a), s2,
                        (100 * self.trans[s][a][s2]/self.count[s][a])
                    ))

"""# 3.Dyna-Q

## 3-1. Q学習クラス(Q_learning)
"""

class Q_learning():
    def __init__(self, env):
        self.env = env  # print用

        self.gamma = 0.9  # 割引率
        self.lr = 0.9     # 学習率
        self.epsilon = 0.1

        self.nb_action = len(env.actions)

        # Q関数
        self.Q = collections.defaultdict(lambda: [0] * self.nb_action)

    def update(self, a_mdp):

        # 近似モデルからランダムにサンプリング
        state, action = a_mdp.samples(1)[0]
        n_state = a_mdp.sample_next_state(state, action)
        reward = a_mdp.get_reward(state, action)
        done = a_mdp.sample_done(state, action)

        #--- Q値の計算
        if done:
            td_error = reward - self.Q[state][action]
        else:
            td_error = reward + self.gamma * max(self.Q[n_state]) - self.Q[state][action]
        self.Q[state][action] += self.lr * td_error


    def sample_action(self, state):
        # ε-greedy
        if np.random.random() < self.epsilon:
            # epsilonより低いならランダムに移動
            return np.random.randint(self.nb_action)
        elif np.sum(self.Q[state]) == 0:
            # Q値が0は全く学習していない状態なのでランダム移動
            return np.random.randint(self.nb_action)
        else:
            # Q値が最大のアクションを実行
            return np.argmax(self.Q[state])

    def print(self):

        # 学習後の Q を出力
        print("Q Q_learning")
        print("----------------------------------------------------")
        for y in range(self.env.H):
            # 上
            s = ""
            for x in range(self.env.W):
                s += "    {:5.2f}    |".format(self.Q[(x,y)][Action.UP.value])
            print(s)
            # 左右
            s = ""
            for x in range(self.env.W):
                s += "{:5.2f}   {:5.2f}|".format(self.Q[(x,y)][Action.LEFT.value], self.Q[(x,y)][Action.RIGHT.value])
            print(s)
            # 下
            s = ""
            for x in range(self.env.W):
                s += "    {:5.2f}    |".format(self.Q[(x,y)][Action.DOWN.value])
            print(s)
            print("-------------------------------------------------------")

"""## 学習"""

env = GridEnv()

a_mdp = A_MDP(env)            # 近似モデル
q_learning = Q_learning(env)  # Q学習

history_reward = []

# 学習ループ
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    # 1episode 最大20stepで終わりにする
    for step in range(20):

        # アクションを決定
        action = q_learning.sample_action(state)

        n_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 近似モデルの学習
        a_mdp.train(state, action, n_state, reward, done)

        # Q学習
        q_learning.update(a_mdp)

        state = n_state
        if done:
            break
    history_reward.append(total_reward)

    interval = 10
    if episode % interval == 0:
        print("train episode {} (min,ave,max)reward {:.1f} {:.1f} {:.1f}".format(
            episode,
            min(history_reward[-interval:]),
            np.mean(history_reward[-interval:]),
            max(history_reward[-interval:]),
        ))

"""## 学習後の近似モデル"""

a_mdp.print()

"""## 学習後のQ関数"""

q_learning.print()

"""## 学習中の報酬の様子"""

# 報酬の推移をグラフで表示
plt.plot(history_reward)
plt.ylim((-2, 1))
plt.grid()
plt.title('training A_MDP')
plt.show()

"""## 100回テストする"""

rewards = []
q_learning.epsilon = 0.01
for episode in range(100):
    state = env.reset()
    #env.render()
    done = False
    total_reward = 0

    # 1episode 最大20stepで終わりにする
    for step in range(20):
        action = q_learning.sample_action(state)
        n_state, reward, done, _ = env.step(action)
        #env.render()
        total_reward += reward

        state = n_state
        if done:
            break
    print("test episode {}  total_reward {}".format(episode, total_reward))
    rewards.append(total_reward)

"""## 100回テストの結果"""

plt.plot(rewards)
plt.ylim((-2, 1))
plt.grid()
plt.title('after train A_MDP')
plt.show()




#-------------------------------------------------------------------------------------------------
"""# 4.Dyna-PolicyIteration

方策反復法

## 4-1. 方策反復法クラス(PolicyIteration)
"""

class PolicyIteration():
    def __init__(self, env):
        self.env = env  # print用

        self.nb_action = len(env.actions)

        self.gamma = 0.9         # 割引率
        self.threshold = 0.0001  # 状態価値関数の計算を打ち切る閾値
        self.epsilon = 0.1

        # 方策関数、最初は均等の確率で初期化
        self.policy = {}
        for s in env.states:
            self.policy[s] = [1/len(env.actions)] * len(env.actions)


    # ポリシーにおける状態の価値を価値反復法で計算
    def estimate_by_policy(self, a_mdp):

        # 状態価値関数の初期化
        V = {s:0 for s in a_mdp.states}

        # 学習
        for i in range(100):  # for safety
            delta = 0

            # 全状態に対して
            for s in a_mdp.states:

                # 各アクションでの報酬期待値を計算
                expected_reward = []
                for a in a_mdp.actions:

                    # 報酬期待値
                    reward = a_mdp.get_reward(s, a)

                    # ポリシーにおけるアクションの遷移確率
                    action_prob = self.policy[s][a]

                    # 割引報酬を計算
                    n_state_probs = a_mdp.get_next_state_probs(s, a)
                    r = 0
                    for s2, s2_prob in n_state_probs.items():
                        # 次の状態の価値を計算
                        r += s2_prob * (reward + self.gamma * V.get(s2, 0))
                    expected_reward.append(action_prob * r)

                # 各アクションの期待値を合計する
                value = sum(expected_reward)
                delta = max(delta, abs(V[s] - value))  # 学習打ち切り用に差分を保存
                V[s] = value

            # 更新差分が閾値以下になったら学習終了
            if delta < self.threshold:
                #print("V count={}".format(i))
                break
        return V

    def update(self, a_mdp):

        # 現policyでの状態価値を計算
        V = self.estimate_by_policy(a_mdp)

        # 学習
        for i in range(100):  # for safety
            update_stable = True

            # 全状態をループ
            for s in a_mdp.states:

                # 各アクションでの報酬期待値を計算
                expected_reward = []
                for a in a_mdp.actions:

                    # 報酬期待値
                    reward =a_mdp.get_reward(s, a)

                    # 割引報酬を計算
                    n_state_probs = a_mdp.get_next_state_probs(s, a)
                    r = 0
                    for s2, s2_prob in n_state_probs.items():
                        # 次の状態の価値を計算
                        r += s2_prob * (reward + self.gamma * V.get(s2, 0))
                    expected_reward.append(r)

                if len(expected_reward) <= 1:
                    continue

                # 期待値が一番高いアクション
                best_action = np.argmax(expected_reward)

                # 現ポリシーで一番選ばれる確率の高いアクション
                policy_action = np.argmax(self.policy[s])

                # ベストなアクションとポリシーのアクションが違う場合は更新
                if best_action != policy_action:
                    update_stable = False

                # ポリシーを更新する
                # ベストアクションの確率を100%にし、違うアクションを0%にする
                for a in a_mdp.actions:
                    if a == best_action:
                        prob = 1
                    else:
                        prob = 0
                    self.policy[s][a] = prob

            # 更新差分が閾値以下になったら学習終了
            if update_stable:
                #print("policy count={}".format(i))
                break


    def sample_action(self, state):
        # ε-Greedy
        if np.random.random() < self.epsilon:
            # epsilonより低いならランダムに移動
            return np.random.randint(self.nb_action)
        else:
            # ポリシー通りに実行
            targets = [i for i in range(self.nb_action)]
            weights = self.policy[state]
            return random.choices(targets, weights=weights, k=1)[0]

    def print_policy(self):

        p = copy.deepcopy(self.policy)
        for s in self.env.states:
            for a in self.env.actions:
                p[s][a] = 100 * p[s][a]

        # 学習後の policy を出力
        print("policy PolicyIteration")
        print("-----------------------------------------------------------")
        for y in range(self.env.H):
            # 上
            s = ""
            for x in range(self.env.W):
                s += "    {:5.1f}%    |".format(p[(x,y)][Action.UP.value])
            print(s)
            # 左右
            s = ""
            for x in range(self.env.W):
                s += "{:5.1f}%  {:5.1f}%|".format(p[(x,y)][Action.LEFT.value], p[(x,y)][Action.RIGHT.value])
            print(s)
            # 下
            s = ""
            for x in range(self.env.W):
                s += "    {:5.1f}%    |".format(p[(x,y)][Action.DOWN.value])
            print(s)
            print("-----------------------------------------------------------")


    def print_value(self, a_mdp):
        # 学習後の policy での V を出力
        print("value")
        V = self.estimate_by_policy(a_mdp)
        for y in range(self.env.H):
            s = ""
            for x in range(self.env.W):
                if (x,y) in V:
                    v = V[(x,y)]
                else:
                    v = 0
                s += "{:5.2f}   ".format(v)
            print(s)

"""## 学習"""

env = GridEnv()

a_mdp = A_MDP(env)             # 近似モデル
policy = PolicyIteration(env)  # 方策反復法

history_reward = []

# 学習ループ
for episode in range(100):
    state = env.reset()
    done = False
    total_reward = 0

    # 1episode 最大20stepで終わりにする
    for step in range(20):

        # アクションを決定
        action = policy.sample_action(state)

        n_state, reward, done, _ = env.step(action)
        total_reward += reward

        # 近似モデルの学習
        a_mdp.train(state, action, n_state, reward, done)

        # ポリシーの学習
        policy.update(a_mdp)

        state = n_state
        if done:
            break
    history_reward.append(total_reward)

    interval = 10
    if episode % interval == 0:
        print("train episode {} (min,ave,max)reward {:.1f} {:.1f} {:.1f}".format(
            episode,
            min(history_reward[-interval:]),
            np.mean(history_reward[-interval:]),
            max(history_reward[-interval:]),
        ))

"""## 学習後の近似モデル"""

a_mdp.print()

"""## 学習後の方策と状態価値関数"""

policy.print_policy()

policy.print_value(a_mdp)

"""## 学習中の報酬の様子"""

# 報酬の推移をグラフで表示
plt.plot(history_reward)
plt.ylim((-2, 1))
plt.grid()
plt.title('training PolicyIteration')
plt.show()

"""## 100回テストする"""

rewards = []
policy.epsilon = 0.01
for episode in range(100):
    state = env.reset()
    #env.render()
    done = False
    total_reward = 0

    # 1episode 最大20stepで終わりにする
    for step in range(20):
        action = policy.sample_action(state)
        n_state, reward, done, _ = env.step(action)
        #env.render()
        total_reward += reward

        state = n_state
        if done:
            break
    print("test episode {} total_reward {}".format(episode, total_reward))
    rewards.append(total_reward)

"""## 100回テストの結果"""

plt.plot(rewards)
plt.ylim((-2, 1))
plt.grid()
plt.title('after PolicyIteration')
plt.show()