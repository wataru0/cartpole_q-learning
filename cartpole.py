#cartpoleを強化学習でやってみる
#Q-Learning,pytorchでDQNでやってみる

import gym 
import numpy as np
from gym import envs
from gym import wrappers
import matplotlib.pyplot as plt
import time
import torch


#線形に等間隔な数列を生成するnumpy.linspace関数
#numpy.linspace(start,stop,num):start-stopまでnum等分する
#観測した状態を離散地に変換する
#Q-Learningでは価値観数の値を離散値で表現する必要ある
def bins(clip_min,clip_max,num):
    return np.linspace(clip_min,clip_max,num+1)[1:-1] #[1:-1]はスライスという表記、要素１から最後の一個前まで


def get_state(observation):
    #各値を6個の離散値に変換
    #np.digitizeは与えられた値をbinsで指定した基数に当てはめる関数。インデックスを返す
    cart_pos,cart_v,pole_angle,pole_v = observation
    digitized = [np.digitize(cart_pos,bins = bins(-2.4,2.4,num_digitized)),
                np.digitize(cart_v,bins=bins(-3.0,3.0,num_digitized)),
                np.digitize(pole_angle,bins=bins(-0.5,0.5,num_digitized)),
                np.digitize(pole_v,bins=bins(-2.0,2.0,num_digitized))]

    #enumerateのforはインデックスとそのオブジェクトを取得することができる
    #**は累乗
    #digitizeは要素数が4の配列
    #6**4=1296通りに分類するので、0~1296に変換
    return sum([x * (num_digitized**i) for i, x in enumerate(digitized)])


#行動を求める関数-----------------------------------------------------------
#ε-greedy法を用いる
def get_action(q_table,next_state,episode):
    #徐々に最適行動のみをとるようにepsilonを設定する
    epsilon = 0.5*(1/(episode+1))
    #epsilon = 0.002

    if np.random.uniform(0,1)>=epsilon:
        next_action = np.argmax(q_table[next_state])

    else:
        next_action = np.random.choice([0,1])

    return next_action

#q-tableを更新する関数-----------------------------------------------------
def update_Qtable(q_table,state,action,reward,next_state):
    gamma = 0.99 #割引率
    alpha = 0.5 #学習率
    next_max_Q = max(q_table[next_state][0],q_table[next_state][1])
    q_table[state][action] = (1-alpha)*q_table[state][action] + alpha*(reward+gamma*next_max_Q)

    return q_table



if __name__ == "__main__":

    #gpuをつかうためのやつ
    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    env = gym.make('CartPole-v0')
    observation = env.reset()
    #結果の出力
    env = wrappers.Monitor(env, './tmp/catrpole',force=True,video_callable=(lambda ep: ep % 100 == 0)) 

    #各パラメータ
    num_digitized = 6
    goal_average_reward = 195
    #Qtableの作成
    q_table = np.random.uniform(low=-1, high=1, size=(num_digitized**4, env.action_space.n))
    total_reward = []


    #確認-------------------
    print(env.observation_space.low)
    print(env.observation_space.high)
    print(observation)

    #メインルーチン
    for episode in range(2000):
        observation = env.reset()
        episode_reward = 0
        
        for step in range(200):
            env.render()
            state = get_state(observation)
            action = get_action(q_table,state,episode) #行動の決定
            next_observation,reward,done,info = env.step(action) #行動による次の状態の決定

            # print("action",action)
            # print("done:",done)
            # print("reward:",reward)

            if done:
                if step <195:
                    reward = -200 #こけたら罰則
                else:
                    reward = 1
            else:
                reward = 1 #各ステップで立ってたら報酬追加

            episode_reward += reward #各episodeでの報酬を格納

            #q_tableの更新
            next_state = get_state(next_observation)
            q_table = update_Qtable(q_table,state,action,reward,next_state)

            observation = next_observation
            state = next_state

            if done: #この文入れないとエラー出る,200step未満で失敗しているのに環境をリセットしていないから
                if episode%100 == 0:
                    print('episode: {},episode_reward: {}'.format(episode,episode_reward))

                total_reward.append(episode_reward)
                break
    #env.colse() #GUI環境の終了

    x = np.arange(0,episode+1,1)
    y = total_reward
    plt.plot(x,y)
    plt.ylim(-200,250)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()


