import random
import numpy as np
from collections import deque
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95                   # 计算未来奖励时的折算率
        self.epsilon = 1.0                  # agent 最初探索环境时选择 action 的探索率
        self.epsilon_min = 0.01             # agent 控制随机探索的阈值
        self.epsilon_decay = 0.995          # 随着 agent 玩游戏越来越好，降低探索率
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(2000, input_dim=self.state_size, activation='relu'))
        model.add(Dense(8000, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    # 根据探索利用原则选择动作
    # 如果产生的随机值小于探索率，随机选择一个动作，否则DQN决策出一个动作
    def act(self, state):
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    # 神经网路利用经验库训练模型，从经验库随机选择batch_size个经验进行训练
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:

            target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 保存模型
    def saveQDNModel(self,saving_path):
        self.model.save(saving_path)
    #装载模型   
    def loadQDNModel(self,loading_path):
        self.model = keras.models.load_model(loading_path)

class TComEnv:
    def __init__(self):
        self.nUEs = 2 #用户数
        self.nCHs = 5 #可分配信道数
        self.trafficPeriod = [10,20] # in TTI
        self.trafficSize = [2000,2000] # in KByte
        self.txPow = np.ones((self.nUEs,1)) #传输功率
        self.noisePow = np.ones((self.nUEs,1))/3 #噪音功率
        #一下参数动态变化，需要在一个新的episode做reset.
        self.state = {
            'buffer_size':2000*np.ones((self.nUEs,)), #kbytes
            'channel_gain': np.ones((self.nCHs,self.nUEs)),
            'new_traffic':np.zeros((self.nUEs,))
        }
        self.iTTI = 1 #传输TTI索引
        self.throughput = 0 #记录累计吞吐量

        self.nActionspace = 2**(self.nUEs*self.nCHs)
        self.observation_space = self.getState()
        
    def reset(self):
        self.throughput = 0
        self.iTTI = 1
        self.state['buffer_size'] = 2000*np.ones((self.nUEs,))
        self.state['channel_gain'] = np.ones((self.nCHs,self.nUEs))
        self.state['new_traffic'] = np.zeros((self.nUEs,))
        return self.getState()


    def step(self,action):

        # 根据动作计算可传输的数据量
        allocated_data_size = self.compute_alloc_data_size(action)
   
        # 评价并给出奖励
        reward, remain_buffer_size = self.compute_reward(allocated_data_size)


        #记录throughput
        self.throughput += sum(self.state['buffer_size'] - remain_buffer_size)

        # 更新缓存
        self.state['buffer_size'] = remain_buffer_size

        # 生成新的信道
        channel = np.random.normal(0, 1, size=(self.nCHs,self.nUEs)) + np.random.normal(0, 1, size=(self.nCHs,self.nUEs))*1j
        self.state['channel_gain']= np.square(np.abs(channel))/2


        # 产生新业务
        self.state['new_traffic'] = np.zeros((self.nUEs,))
        for iUE in range(self.nUEs):
            if self.iTTI % self.trafficPeriod[iUE] == 0:
                self.state['new_traffic'][iUE] = self.trafficSize[iUE]

        self.iTTI += 1 #更新计数器
        
        new_state = self.getState()
        
        # 更新缓存
        self.state['buffer_size'] += self.state['new_traffic']
        return new_state, reward




    def compute_alloc_data_size(self,action):


        #把动作编号转变为二进制
        alloc_arr = np.zeros((self.nUEs*self.nCHs,))
        alloc_list = [int(x) for x in bin(action)[2:]]
        alloc_arr[-len(alloc_list):] = alloc_list

        #把二进制串变形成为nUEs X nCHs的分配矩阵
        alloc_arr = alloc_arr.reshape((self.nCHs, self.nUEs))


        #信号功率矩阵
        des_pow_arr = np.multiply(alloc_arr,self.state['channel_gain'])
        des_pow_arr = np.multiply(des_pow_arr, self.txPow.reshape((1,2)))

        #每个信道上的信号总强度
        ch_pow_arr = np.multiply(alloc_arr,self.state['channel_gain'])
        ch_pow_arr = np.matmul(ch_pow_arr,self.txPow)
        #alternativly
        #ch_pow_arr = np.sum(des_pow_arr,axis=1).reshape((nCHs,1))

        #干扰矩阵
        int_pow_arr = ch_pow_arr - des_pow_arr


        #信噪干扰比矩阵
        sinr_arr = np.divide(des_pow_arr,int_pow_arr + self.noisePow.reshape((1,2)))

        #可传输数据量矩阵
        bytes_arr = 180*np.log2(1 + sinr_arr)/8

        #每个用户分配到的可传输数据量
        alloc_data_size= np.sum(bytes_arr, axis = 0)

        return alloc_data_size


    def compute_reward(self,alloc_data_size):

            #奖励分两部分，1）针对UE的 2）公平奖励
            #针对UE的，分配的容量跟缓存的容量越接近越好
            UE_reward = np.zeros((2,))

            for i in range(self.nUEs):
                if self.state['buffer_size'][i] == 0:
                    UE_reward[i] = - alloc_data_size[i]
                else:
                    UE_reward[i] = - abs(self.state['buffer_size'][i] - alloc_data_size[i])/self.state['buffer_size'][i]

            #公平奖励，用户的缓存余量越接近越好
            remain_buffer_size = self.state['buffer_size']  - alloc_data_size
            remain_buffer_size = remain_buffer_size.clip(min = 0)

            if np.max(remain_buffer_size)-np.min(remain_buffer_size) < 1: 
                fair_reward = 0
            else:
                fair_reward = - (np.max(remain_buffer_size)-np.min(remain_buffer_size))/np.max(remain_buffer_size)

            reward = sum(UE_reward) + fair_reward

            return reward, remain_buffer_size

    def getThroughput(self):
        return self.throughput

    def getState(self):
        new_buffer_size = self.state['buffer_size'] + self.state['new_traffic']
        return np.concatenate((new_buffer_size, self.state['channel_gain'].reshape((self.nUEs*self.nCHs,))))

    def getBufferSize(self):
        return list(self.state['buffer_size'])

	
    
if __name__ == "__main__":
    # 开始迭代
    for e in range(EPISODES):
    
        # 每次开始时都重新设置一下状态
        state = env.reset()

        state = np.reshape(state, [1, state_size])
        
        # time 代表每一个传输的TTI，
        progress_bar = tqdm(list(range(TTIs)))
        for time in progress_bar:
        #for time in range(1000):
            # 每一TTI时，agent 根据 state 选择 action
            action = agent.act(state)
            # 这个 action 使得进入下一个状态 next_state，并且拿到了奖励 reward
            next_state, reward  = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            
            # 记忆之前的信息：state, action, reward
            agent.remember(state, action, reward, next_state)
            
            # 更新下一所在状态
            state = next_state
            
            # 用之前的经验训练 agent   
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


            progress_bar.set_description("episode: {}/{}, TTI: {}/{}".format(e+1, EPISODES, time+1, TTIs))

        #每1000个TTI打印一下进展    
        print("episode: {}/{}, throughput: {}, buffer: {} epsilon: {:.2}"
                      .format(e+1, EPISODES, int(env.getThroughput()), 
                        [int(i) for i in env.getBufferSize()], agent.epsilon))
        throughputThread[e] = env.getThroughput()
    agent.saveQDNModel('saved_model')
