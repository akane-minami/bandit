import random
import numpy as np

class Softmax():
    def __init__(self,tau, counts, values): 
        self.counts = counts  # armの引く回数
        self.values = values  # 引いたarmから得られた報酬の平均値
        self.tau=tau

    def initialize(self, n_arms): 
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self):

        n_arms = len(self.counts)
        for arm in range(n_arms):#最初にどのarmも1回ずつ選ぶ
            if self.counts[arm] == 0:
                return arm

        exp_values = [0.0 for arm in range(n_arms)]
        exp_values=np.exp(np.array(self.values)/self.tau)#分子
        sum_exp_value=sum(exp_values)#分母
        prb=np.array(exp_values)/sum_exp_value#armの探索確率


        return int(np.random.choice(n_arms,1,p=prb))#n_armsの中からprbの確率でひく

    def update(self, chosen_arm, reward):#報酬の更新
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1 
        n = self.counts[chosen_arm] # 今回のアームを選択した回数
        value = self.values[chosen_arm]  # 更新前の平均報酬額
        new_value = ((n-1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value