import numpy as np
class TComEnv:
    def __init__(self):
    	self.nUEs = 2
    	self.nCHs = 5
    	self.iTTI = 0
    	self.trafficPeriod = [10,20] # in TTI
    	self.trafficSize = [100,50] # in Byte
    	self.txPow = np.ones((self.nUEs,))
    	self.noisePow = np.ones((self.nUEs,))/3
    	self.state = self.reset()

    def reset(self):
    	state.buffer_size = 100*np.ones((self.nUEs,))
    	state.channel_gain = np.ones((self.nCHs,self.UEs))
    	state.new_traffic = np.zeros((self.nUEs,))
    	return state

    def setp(self,action):

    	# 根据动作计算可传输的数据量
    	allocated_data_size = self.compute_alloc_data_size(action)

    	#计算传输后的缓存容量
    	state.buffer_size = min(self.state.buffer_size - allocated_data_size, 0)

    	# 评价并给出奖励
    	reward = self.compute_reward(allocated_data_size)


    	# 生成新的信道
    	channel = np.random.normal(0, 1, size=(self.nCHs,self.nUEs)) + np.random.normal(0, 1, size=(self.nCHs,self.nUEs))*j
    	state.channel_gain = np.square(channel)/2

    	# 产生新业务
    	state.new_traffic = np.zeros((self.nUEs,))
    	for iUE in range(1,self.nUEs)
    		if self.iTTI % self.trafficPeriod(i) == 0
    			state.new_traffic(i) = self.trafficSize(i)






    def compute_alloc_data_size(self,action):


    	#把动作编号转变为二进制
    	alloc_vec = np.array([int(x) for x in bin(action)[2:]])

    	#把二进制串变形成为nUEs X nCHs的分配矩阵
    	alloc_mtx = alloc_vec.reshape(self.nCHs,self.nUEs)

    	#可传输的数据量
    	allocated_data_size = np.zeros((self.nUEs,))

    	# INR
    	for iUE in range(1,nUEs):
    		for iCH in range(1,nCHs):
    			SINR = alloc_mtx(iCH,iUE) * self.state.channel_gain(iCH,iUE)*self.txPow(iUE)/self.noisePow(iUE)
    			allocated_volme(iUE) += 180000 * 




    def compute_reward(self,allocated_data_size):
    	


