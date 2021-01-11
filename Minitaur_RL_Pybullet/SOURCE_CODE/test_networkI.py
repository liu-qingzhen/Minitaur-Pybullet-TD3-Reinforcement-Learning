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


input_dims = [28]
n_actions=8
env_id = "MinitaurBulletEnv-v0"
load_path = os.path.join(os.getcwd(),'Trained_model_IV','actor_td3')

# Because I saved the parameters instead of the model
# We need to initiate an actor network to load the model
class Network_I(nn.Module):
    def __init__(self, lr, input_dims, n_fc1, n_fc2,
            n_actions, name):
        super(Network_I, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = n_fc1
        self.fc2_dims = n_fc2
        self.n_actions = n_actions
        self.name = name

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        actions = self.fc1(state)
        actions = F.relu(actions)
        actions = self.fc2(actions)
        actions = F.relu(actions)

        actions = T.tanh(self.mu(actions))

        return actions

    def load(self,load_path):
        print('...Loading Trained Model...')
#       If you want to run with cpu, then the map_location=='cpu' is neccesary
#       because I trained the model with gpu
        self.load_state_dict(T.load(load_path,map_location='cpu'))

#   Choose actions based on observations
#   We don't need warmup in testing actually, so it can be set to 0
def choose_action(observation,act_net,noise,time_step,warmup=0):
    if time_step < warmup:
        action = T.tensor(np.random.normal(scale=0.4,
                                       size=(n_actions,)))
    else:
        observation = T.tensor(observation,dtype=T.float).to(act_net.device)
        action = act_net.forward(observation).to(act_net.device)
    action_ = T.tensor(action+noise,dtype=T.float).to(act_net.device)
    action_ = T.clamp(action_,-1,1)
    return action_.cpu().detach().numpy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",default=False,action='store_true',help='Enable CUDA True or False')
    parser.add_argument("--name","--name",required=False,default='5406',help="Name of the run")
    args = parser.parse_args()
    device = T.device("cuda:0" if args.GPU else "cpu")
    load_path = load_path
    # os.makedirs(load_path,exist_ok=True)

    writer = SummaryWriter(comment="-TD3" + args.name)


#   just the normal gym process

    env = gym.make(env_id)
    act_net = Network_I(lr=3e-4,input_dims=input_dims,n_fc1=400,n_fc2=300,n_actions=8,name='actor')
    act_net.load(load_path)
    scores = []
    time_step = 0
    best_score=0.2
    for i in range(50):
        obs = env.reset()
        finished = False
        score = 0
        while not finished:
            act = choose_action(obs,act_net,noise=np.random.normal(scale=0.2),time_step=time_step,warmup=0)
            time_step+=1
            act = np.clip(act,-1,1)
            # print('log1')
            state_,reward,finished,_ = env.step(act)
            obs = state_
            score += reward
        writer.add_scalar('Score_for_each_episode',score,time_step)
        scores.append(score)
        average_score = np.mean(scores[-50:])
        print('---------------------------------------------------------')
        if score<4:
            print('····Score is low, Its better to replace the default environment file with mine····')
            print('····There is a video showing where to replace in the project file····')
        print('episode:',i)
        print('total_steps:%d'%time_step)
        print('Average score for episode',i,' is %.2f'%score)
        print('Average score for last 50 episodes is %.2f'%average_score)
        print('---------------------------------------------------------')


    writer.close()