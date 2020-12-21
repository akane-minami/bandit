import random
import numpy as np
import matplotlib.pyplot as plt

from algo.epsilon_greedy import Epsilon_Greedy
from algo.softmax import Softmax

class BernoulliArm():
    """ベルヌーイバンディット"""
    def __init__(self,p):
      self.p=p
    def drow(self):
      """報酬確率に従って0/1を返す"""
      if random.random()>self.p:
        return 0.0
      else:
        return 1.0

def test_algorithm(algo,arms,n_sims,n_steps,theta):
  """シミュレーションを実行
    Args:
     algo:指定された方策を持つAgent
     arms:各報酬確率を持つ腕
     n_sims(int):シミュレーション数
     n_steps(int):ステップ数

    Returns:
     times:step数
     chosen_arms:各stepで選択した腕
     rewards:各stepの報酬
     cumulative_rewards:各stepまでの累積報酬
     regrets:regret(累積)

  """
  #結果表示用の配列を初期化
  chosen_arms = np.zeros((n_sims,n_steps))
  rewards = np.zeros((n_sims,n_steps))
  cumulative_rewards = np.zeros((n_sims,n_steps))
  times = np.zeros((n_sims,n_steps))
  regrets= np.zeros((n_sims,n_steps))

  for sim in range(n_sims):
    algo.initialize(len(arms))#Agent初期化
    theta_reg=np.zeros(n_steps)
    for step in range(n_steps):
      chosen_arm=algo.select_arm()#腕を方策に従って決める
      reward=arms[chosen_arm].drow()#腕を引いて報酬を得る
      #結果を配列に格納
      times[sim,step] += step
      chosen_arms[sim,step]+=chosen_arm
      rewards[sim,step] =reward
      if step==1:
        cumulative_rewards[sim,step] = reward
      else:
        cumulative_rewards[sim,step] =cumulative_rewards[sim,step-1] + reward
      theta_reg[step]+=theta[chosen_arm]
      regrets[sim,step] +=np.max(theta)*step - np.sum(theta_reg)
      algo.update(chosen_arm,reward)#方策の更新
  return [times, chosen_arms, rewards, cumulative_rewards,regrets]

def run(algo,label,n_sims,n_steps,n_arms,arms,theta):
  print(label)
  algo.initialize(n_arms)
  results=test_algorithm(algo,arms,n_sims,n_steps,theta)
  y=np.mean(results[4],axis=0)#平均
  plt.plot(np.linspace(1, n_steps, num=n_steps),y , label=label) #平均して出力
  plt.legend(loc="best")
  plt.xlabel("step")
  plt.ylabel("regret")

def main():
  n_sims = 100
  n_steps = 1000

  #腕の生成
  theta=np.array([0.3,0.5,0.7])
  n_arms=len(theta)
  random.shuffle(theta)
  arms = map(lambda x: BernoulliArm(x), theta)
  arms = list(arms)

  algo = Epsilon_Greedy(0.3,[],[])#指定した方策をもつAgent作成
  run(algo,"ε-greedy ε=0.3",n_sims,n_steps,n_arms,arms,theta)
  algo = Softmax(0.1,[],[])
  run(algo,"softmax tau=0.1",n_sims,n_steps,n_arms,arms,theta)

main()
plt.show()

