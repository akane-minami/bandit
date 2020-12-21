import random
import numpy as np

class Epsilon_Greedy():
    def __init__(self,epsilon,count,value):
      self.epsilon = epsilon
      self.counts = count
      self.values = value

    def initialize(self,n_arms):
      self.counts=np.zeros(n_arms)
      self.values=np.zeros(n_arms)

    def select_arm(self):
      if random.random() > self.epsilon:
        return np.random.choice([arm_ for arm_, greedy in enumerate(self.values) if greedy == np.max(self.values)])
      else:
        return random.randrange(len(self.values))

    def update(self,chosen_arm,reward):
      #価値更新
      self.counts[chosen_arm]+=1
      n=self.counts[chosen_arm]
      value=self.values[chosen_arm]
      self.values[chosen_arm]=((n-1) / float(n)) * value + (1 / float(n)) * reward
