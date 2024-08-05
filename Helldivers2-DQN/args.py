from utils import calculate_eps_decay

class Args():
    def __init__(self):
        # Network Hyperparameters
        self.BUFFER_SIZE = int(1e5)      # Replay buffer size
        self.BATCH_SIZE = 50             # Minibatch size
        self.TAU = 1e-3                  # Soft update coefficient for target parameters
        self.LR = 5e-4                   # Learning rate
        self.UPDATE_EVERY = 2            # Network update frequency
        
        # Model Training Settings
        self.seed = 0                    # Random seed
        self.reward = 10                 # Reward for correct prediction
        self.max_steps = 500000          # Maximum number of training steps
        self.test_steps = 100            # Number of steps to evaluate the model
        self.num_labels = 4              # Number of different labels in the dataset
        self.Double_DQN = True           # Whether to use Double-DQN
        self.priority_replay = [0.5, 0.5, 0.5]  # Priorities for replay buffer

        # Exploration Settings
        self.eps_start = 1.0             # Starting value of epsilon for epsilon-greedy action selection
        self.eps_end = 0.01              # Minimum value of epsilon
        self.eps_percentage = 0.98       # Percentage of max_steps over which to decay epsilon
        self.eps_decay = calculate_eps_decay(self.eps_start, self.eps_end, self.max_steps, self.eps_percentage)
        
        # Environment and Image Settings
        self.image_dir = 'ImageProcessing/TestingData/Filtered/'  # Directory containing images
        self.policy_file = 'learned_policy_3.pth'
        self.image_w = 25                # Image width after processing
        self.image_h = 25                # Image height after processing
        self.image_size = self.image_w * self.image_h  # Image size for processing
        
        # Miscellaneous Settings
        self.wandb = False                # Whether to log to Weights & Biases
        self.load_policy = True         # Whether to load a saved policy
