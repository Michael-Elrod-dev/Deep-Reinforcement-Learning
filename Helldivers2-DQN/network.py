import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
import torch
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Network():
    def __init__(self, args):
        self.image_size = args.image_size
        self.num_labels = args.num_labels
        self.seed = random.seed(args.seed)
        self.BUFFER_SIZE = BUFFER_SIZE
        # setting optional extra techniques
        self.Double_DQN = args.Double_DQN
        self.prio_e, self.prio_a, self.prio_b = args.priority_replay

        # Q-Network
        self.qnetwork_local = QNetwork(args.image_size, args.num_labels, args.seed).to(device)
        self.qnetwork_target = QNetwork(args.image_size, args.num_labels, args.seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(args.num_labels, BUFFER_SIZE, BATCH_SIZE, args.seed, args.Double_DQN, args.priority_replay)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, image, label, reward, next_image, done):
        # Save experience in replay memory
        self.memory.add(image, label, reward, next_image, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences, experience_indexes, priorities = self.memory.sample()
                self.learn(experiences, experience_indexes, priorities, GAMMA)

    def act(self, image, eps=0.):
        image = image.float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            label_values = self.qnetwork_local(image)
        self.qnetwork_local.train()

        # Epsilon-greedy label selection
        if random.random() > eps:
            return np.argmax(label_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.num_labels))

    def learn(self, experiences, experience_indexes, priorities, gamma):
        images, labels, rewards, next_images, dones = experiences

        ## compute and minimize the loss
        # calculate current Q_sa
        Q_s = self.qnetwork_local(images)
        Q_s_a = self.qnetwork_local(images).gather(1, labels)


        # Get max predicted Q values (for next images) from target model
        if self.Double_DQN:
            # double DQN uses the local network for selecting best label and evaluates it with target network
            best_labels = self.qnetwork_local(next_images).max(1)[1].unsqueeze(1)
            Q_s_next = self.qnetwork_target(next_images).gather(1, best_labels)
        else:
            Q_s_next = self.qnetwork_target(next_images).max(1)[0].unsqueeze(1)
            
        targets = rewards + gamma * Q_s_next * (1 - dones)
        
        # calculate loss between the two
        losses = (Q_s_a - targets)**2
        
        # importance-sampling weights aka formula from Prioritized Experience Replay
        importance_weights = (((1/self.BUFFER_SIZE)*(1/priorities))**self.prio_b).unsqueeze(1)
        
        loss = (importance_weights*losses).mean()

        # calculate gradients and do a step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # calculate priorities and update them
        target_priorities = abs(Q_s_a - targets).detach().cpu().numpy() + self.prio_e
        self.memory.update_priority(experience_indexes, target_priorities)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def update_beta(self, interpolation):
        self.prio_b += (1 - self.prio_b)*interpolation


class ReplayBuffer:
    def __init__(self, num_labels, buffer_size, batch_size, seed, Double_DQN, Priority_Replay_Paras):
        self.num_labels = num_labels
        self.memory = deque(maxlen=buffer_size)  
        self.priority = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["image", "label", "reward", "next_image", "done"])
        self.seed = random.seed(seed)
        self.prio_e, self.prio_a, self.prio_b = Priority_Replay_Paras
    
    def add(self, image, label, reward, next_image, done):
        e = self.experience(image, label, reward, next_image, done)
        self.memory.append(e)
        self.priority.append(self.prio_e)
        
    def update_priority(self, priority_indexes, priority_targets):
        for index,priority_index in enumerate(priority_indexes):
            self.priority[priority_index] = priority_targets[index][0]
    
    def sample(self):
        adjusted_priority = np.array(self.priority)**self.prio_a
        sampling_probability = adjusted_priority/sum(adjusted_priority)
        experience_indexes = np.random.choice(np.arange(len(self.priority)), size=self.batch_size, replace=False, p=sampling_probability)
        experiences = [self.memory[index] for index in experience_indexes]

        images = torch.stack([e.image for e in experiences if e is not None]).to(device)
        labels = torch.from_numpy(np.vstack([e.label for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_images = torch.stack([e.next_image for e in experiences if e is not None]).to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.array([self.priority[index] for index in experience_indexes])).float().to(device)

        return (images, labels, rewards, next_images, dones), experience_indexes, priorities

    def __len__(self):
        return len(self.memory)
    