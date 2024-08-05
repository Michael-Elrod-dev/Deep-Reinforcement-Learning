import torch
import numpy as np
from args import Args
from logger import Logger
from network import Network
from collections import deque
from utils import *


def test_dqn(images):
    args = Args()
    network = Network(args)
    network.qnetwork_local.load_state_dict(torch.load(args.policy_file))
    scores = []
    eps = 0
    step = 0
    
    for image in images:
        step += 1
        
        # Process the image
        processed_image = preprocess_image(image, args.image_h, False)
        processed_image = torch.flatten(processed_image)
        
        predicted_label = network.act(processed_image, eps)

        # Convert the predicted label to the corresponding key press action
        if predicted_label == 0:
            print("Left")
            APress()
        elif predicted_label == 1:
            print("Right")
            DPress()
        elif predicted_label == 2:
            print("Up")
            WPress()
        elif predicted_label == 3:
            print("Down")
            SPress()

    return scores

def dqn(args, logger):
    network = Network(args)
    scores = []
    high_score = 0
    eps = args.eps_start
    scores_window = deque(maxlen=100)

    for step in range(1, args.max_steps + 1):
        reward = 0
        # Get an image path and its label at random
        image_path, true_label = get_random_image(args.image_dir)
        
        # Process the image
        processed_image = preprocess_image(image_path, args.image_h, False)
        processed_image = torch.flatten(processed_image)

        network.update_beta((step - 1) / (args.max_steps - 1))

        # Send to and update the network
        predicted_label = network.act(processed_image, eps)
        next_image, reward, done = env_step(args, predicted_label, true_label, processed_image)
        network.step(processed_image, predicted_label, reward, next_image, done)
        
        scores_window.append(reward)
        scores.append(reward)

        if logger is not None and step % 1000 == 0:
            avg_score = np.mean(scores_window)
            logger.log_metrics(eps, avg_score)
            print('\rStep: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(step, eps, reward, avg_score))
        else:
            print('\rStep: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(step, eps, reward, np.mean(scores_window)), end="")

        if eps > args.eps_end:
            eps -= args.eps_decay
        else:
            eps = args.eps_end

        if np.mean(scores_window) > high_score:
            high_score = np.mean(scores_window)
            torch.save(network.qnetwork_local.state_dict(), 'checkpoint.pth')
        
    torch.save(network.qnetwork_local.state_dict(), 'checkpoint.pth')

    return scores
