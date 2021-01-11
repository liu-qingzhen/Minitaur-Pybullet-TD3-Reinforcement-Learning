import numpy as np
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import gym
import argparse
import pybullet_envs
from torch.utils.tensorboard import SummaryWriter
import copy

input_dims = [28]
n_actions=8
batch_size=128
env_id = "MinitaurBulletEnv-v0"
gamma=0.99
tau = 0.005
update_actor =2
replay_initial = 1000
warmup = 1000
folder_path = os.getcwd()


# I referred to the code structure of Learning Machine Learning with Phil on Youtube
# and the TD3 paper code at the beginning. Then I programmed my own code.



# create a replay buffer to save the transitions
# improve the sample efficiency
class ReplayBuffer():
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.new_state_buffer = np.zeros((self.buffer_size, *input_shape))
        self.action_buffer = np.zeros((self.buffer_size, n_actions))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.terminal_buffer = np.zeros(self.buffer_size, dtype=np.bool)

#   store the information for after each action taken

    def store(self, state, action, reward, state_, done):
        index = self.buffer_counter % self.buffer_size
        self.state_buffer[index] = state
        self.new_state_buffer[index] = state_
        self.terminal_buffer[index] = done
        self.reward_buffer[index] = reward
        self.action_buffer[index] = action

        self.buffer_counter += 1


#   sample a bath of data for updating network parameters
    def sample_batch(self, batch_size):
        max_mem = min(self.buffer_counter, self.buffer_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_buffer[batch]
        states_new = self.new_state_buffer[batch]
        actions = self.action_buffer[batch]
        rewards = self.reward_buffer[batch]
        dones = self.terminal_buffer[batch]

        return states, actions, rewards, states_new, dones

# create the critic network structure
class Critic_Network(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2, n_actions,
            name, chkpt_dir=''):
        super(Critic_Network, self).__init__()
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fc1_dims = n_fc1
        self.fc2_dims = n_fc2

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
#  forward to generate critic value with input of states and actions
    def forward(self, state, action):
        q = self.fc1(T.cat([state, action], dim=1))
        q = F.relu(q)
        q = self.fc2(q)
        q = F.relu(q)

        q = self.q(q)

        return q

#   saving the parameters of the model and load the dict is 
#   faster than directly saving and loading model
    def save(self,save_path):
        T.save(self.state_dict(),os.path.join(save_path,self.name+'_td3'))

    def load(self,load_path):
        self.load_state_dict(T.load(os.path.join(save_path,self.name+'_td3')))

#   create the actor network structure
class Actor_Network(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2,
            n_actions, name):
        super(Actor_Network, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = n_fc1
        self.fc2_dims = n_fc2
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
  
#   forward to generate actions with input of states
    def forward(self, state):
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)

        actions = T.tanh(self.action(actions))

        return actions

    def save(self,save_path):
        T.save(self.state_dict(), os.path.join(save_path,self.name+'_td3'))

    def load(self,load_path):
        self.load_state_dict(T.load(os.path.join(load_path,self.name+'_td3')))
#   create the agent
class agent(object):
    def __init__(self,actor,critic1,critic2,target_actor,target_critic1,target_critic2,replay_buffer,noise,tau,gamma,batch_size):
        self.actor_network = actor
        self.critic_network1 = critic1
        self.critic_network2 = critic2
        self.target_actor = target_actor
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.tau = tau
        self.gamma = gamma
        self.noise = noise

    def save_to_buffer(self, state, action, reward, new_state, done):
        self.replay_buffer.store(state, action, reward, new_state, done)

#   transfer data to float tensors 
def toTensor(name,net):
    name = T.tensor(name,dtype=T.float).to(net.device)
    return name

#   get the parameters of a net 
def getParams(net):
    return dict(net.named_parameters())

#   to choose action. there will be a warm up period in first training
#   I think this operation is for the initiation of network parameters
def choose_action(observation,act_net,noise,time_step,warmup=1000):
    if time_step < warmup:
        action_ = T.tensor(np.random.normal(scale=0.4,
                                       size=(n_actions,)))

#   after that period, agent will choose action based on network output + noises
    else:
        observation = T.tensor(observation,dtype=T.float).to(act_net.device)
        action = T.tensor(act_net.forward(observation),dtype=T.float).to(act_net.device)
        action_ = T.tensor(action+noise,dtype=T.float).to(act_net.device)
        action_ = T.clamp(action_,-1,1)
    return action_.cpu().detach().numpy()

#   save all the network to save path
def save(act,cri1,cri2,tgt_act,tgt_cri1,tgt_cri2,save_path):
    print('... saving model ...')
    act.save(save_path)
    cri1.save(save_path)
    cri2.save(save_path)
    tgt_act.save(save_path)
    tgt_cri1.save(save_path)
    tgt_cri2.save(save_path)

#   load all the network needed for training
def load(act,cri1,cri2,tgt_act,tgt_cri1,tgt_cri2,load_path):
    print('... loading model ...')
    act.load(load_path)
    cri1.load(load_path)
    cri2.load(load_path)
    tgt_act.load(load_path)
    tgt_cri1.load(load_path)
    tgt_cri2.load(load_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#   default is to train with cpu, but gpu is recommended for this project
    parser.add_argument("--GPU",default=False,action='store_true',help='Enable CUDA True or False')
#   name of folder to save and load models
    parser.add_argument("--name","--name",required=False,default='5406_Model',help="Name of the run")
    args = parser.parse_args()
#   pytorch require us to move tensors to device
    device = T.device("cuda:0" if args.GPU else "cpu")
#   path to save model
    save_path = os.path.join(folder_path,args.name)
#   It's recommended to use the absolute path in case something wrong with this relative path
    os.makedirs(save_path,exist_ok=True)
#   create the writer for logging with tensorboard
    writer = SummaryWriter(comment="5406_Project_Checking")
#   generate environment
    env = gym.make(env_id)
#   initiate networks
    act_net = Actor_Network(lr=3e-4,input_dims=input_dims,n_fc1=256,n_fc2=256,n_actions=8,name='actor')
    cri_net1 = Critic_Network(lr=3e-4,input_dims=input_dims, n_fc1=256, n_fc2=256, n_actions=8, name='critic1')
    cri_net2 = Critic_Network(lr=3e-4,input_dims=input_dims, n_fc1=256, n_fc2=256, n_actions=8, name='critic2')

    target_act = Actor_Network(lr=3e-4,input_dims=input_dims,n_fc1=256,n_fc2=256,n_actions=8,name='target_actor')
    target_cri1 = Critic_Network(lr=3e-4,input_dims=input_dims, n_fc1=256, n_fc2=256, n_actions=8, name='target_critic1')
    target_cri2 = Critic_Network(lr=3e-4,input_dims=input_dims, n_fc1=256, n_fc2=256, n_actions=8, name='target_critic2')

#   This will also work but the name will be the same so we won't have 6 files of savings
    # cri_net2 = copy.deepcopy(cri_net1)
    # target_act = copy.deepcopy(act_net)
    # target_cri1 = copy.deepcopy(cri_net1)
    # target_cri2 = copy.deepcopy(cri_net1)
    print(act_net)
    print(cri_net1)

#   initiate the replay buffer
    replay_buffer = ReplayBuffer(buffer_size=10000,input_shape=[28],n_actions=8)
#   noise for choosing actions
    noise = np.random.normal(scale=0.2)
#   initiate the agent
    agent = agent(actor=act_net,critic1=cri_net1,critic2=cri_net2,target_actor=target_act,target_critic1=target_cri1,target_critic2=target_cri2,replay_buffer=replay_buffer,noise=noise,tau=1,gamma=gamma,batch_size=batch_size)
#   for recording scores of each episode
    scores = []
#   total steps
    time_step = 0
#   for saving model
    best_score=-1
    for i in range(2000):
        obs = env.reset()
        finished = False
        score = 0
        while not finished:
            act = choose_action(obs,act_net,noise,time_step=time_step,warmup=warmup)
            time_step+=1
            act = np.clip(act,-1,1)
            # print('log1')
            state_,reward,finished,_ = env.step(act)
            obs = state_
            score += reward
            # writer.add_scalar("reward",reward,time_step)
            agent.save_to_buffer(obs,act,reward,state_,finished)
#           start updating parameters
            if replay_buffer.buffer_counter > max(batch_size,replay_initial):
                state,action,reward,new_state,done = replay_buffer.sample_batch(batch_size)
#               for pytorch
                reward = toTensor(reward,cri_net1)
                done = T.tensor(done).to(cri_net1.device)
                action = toTensor(action,cri_net1)
                state = toTensor(state,cri_net1)
                new_state = toTensor(new_state,cri_net1)

#               create target actions with noises
                target_actions = target_act.forward(new_state)
                target_actions = target_actions + \
                                 T.clamp(T.tensor(np.random.normal(scale=0.2)),-0.5,0.5)
                target_actions = T.clamp(target_actions,-1,1)

#               get the state-action value from 2 critic network and 2 target critic networks
                q1_ = target_cri1.forward(new_state,target_actions)
                q2_ = target_cri2.forward(new_state,target_actions)
                q1 = cri_net1.forward(state,action)
                q2 = cri_net2.forward(state,action)

#               as the formula shows Q = R if it is done; or Q = R + gamma * Q'
                q1_[done] = 0.0
                q2_[done] = 0.0

                q1_ = q1_.view(-1)
                q2_ = q2_.view(-1)

                q_ = T.min(q1_,q2_)

                target = reward + agent.gamma * q_
                target = target.view(batch_size,1)

#               update critic network
                cri_net1.optimizer.zero_grad()
                cri_net2.optimizer.zero_grad()
                q1_loss = F.mse_loss(target,q1)
                q2_loss = F.mse_loss(target,q2)
                critic_loss = q1_loss + q2_loss
                critic_loss.backward()
                cri_net1.optimizer.step()
                cri_net2.optimizer.step()

#               update actor network and target networks every fixed episodes. 
#               One feature of TD3. This is what is called DELAYED
                if time_step % update_actor == 0:
                    act_net.optimizer.zero_grad()
                    actor_q1_loss = cri_net1.forward(state,act_net.forward(state))
                    actor_loss = -T.mean(actor_q1_loss)
                    actor_loss.backward()
                    act_net.optimizer.step()

#                   then we should update the target networks 
                    for param, target_param in zip(act_net.parameters(), target_act.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(cri_net1.parameters(), target_cri1.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                    for param, target_param in zip(cri_net2.parameters(), target_cri2.parameters()):
                        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)



#    Below are the complicated codes for updating parameters
#    Just ignore it

                    # actor_params = act_net.named_parameters()
                    # critic_params1 = cri_net1.named_parameters()
                    # critic_params2 = cri_net2.named_parameters()
                    # target_actor_params = target_act.named_parameters()
                    # target_critic_params1 = target_cri1.named_parameters()
                    # target_critic_params2 = target_cri1.named_parameters()

                    # critic_state_dict1 = dict(critic_params1)
                    # critic_state_dict2 = dict(critic_params2)
                    # actor_state_dict = dict(actor_params)
                    # target_critic_dict1 = dict(target_critic_params1)
                    # target_critic_dict2 = dict(target_critic_params2)
                    # target_actor_dict = dict(target_actor_params)
                    # for name in critic_state_dict1:
                    #     critic_state_dict1[name] = tau * critic_state_dict1[name].clone() + \
                    #                               (1 - tau) * target_critic_dict1[name].clone()
                    # target_cri1.load_state_dict(critic_state_dict1)

                    # for name in critic_state_dict2:
                    #     critic_state_dict2[name] = tau * critic_state_dict2[name].clone() + \
                    #                               (1 - tau) * target_critic_dict2[name].clone()
                    # target_cri2.load_state_dict(critic_state_dict2)

                    # for name in actor_state_dict:
                    #     actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                    #                              (1 - tau) * target_actor_dict[name].clone()
                    # target_act.load_state_dict(actor_state_dict)
        

#       Because there will not be critic loss before replay initial
#       and no actor loss when in the delayed steps
#       so we can only write these data after such steps
        if time_step>max(replay_initial,warmup)+update_actor:
            writer.add_scalar('q1_loss',q1_loss,time_step)
            writer.add_scalar('q2_loss',q2_loss,time_step)
            writer.add_scalar('actor_loss',actor_loss,time_step)
            writer.add_scalar('critic_loss',critic_loss,time_step)
        writer.add_scalar('Score_for_each_episode',score,time_step)
        scores.append(score)
        average_score = np.mean(scores[-50:])
#       update the best_score, so we can save the best model
        if score > best_score:
            best_score = score
            save(act_net,cri_net1,cri_net2,target_act,target_cri1,target_cri2,save_path=save_path)
        print('total_steps:%d'%time_step)
        print('score for episode ',i,' is %.2f'%score)
        print('average socre:',np.mean(scores[-100:]))


    writer.close()