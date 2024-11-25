import gymnasium as gym
import numpy as np
import torch
from ppo_agent import PPOAgent
import matplotlib.pyplot as plt
import os
import time

def create_save_directory(dir_name='saved_models'):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def save_model(agent, save_dir, prefix='', is_best=False):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if is_best:
        actor_filename = os.path.join(save_dir, f'{prefix}best_actor.pth')
        critic_filename = os.path.join(save_dir, f'{prefix}best_critic.pth')
    else:
        actor_filename = os.path.join(save_dir, f'{prefix}actor_{timestamp}.pth')
        critic_filename = os.path.join(save_dir, f'{prefix}critic_{timestamp}.pth')
    
    torch.save(agent.actor.state_dict(), actor_filename)
    torch.save(agent.critic.state_dict(), critic_filename)
    #print(f"Saved {'best ' if is_best else ''}model to {actor_filename}")

def plot_learning_curve(scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(scores, label='Episode Scores')
    plt.plot(running_avg, label='Running Average')
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(figure_file)
    plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    print(f"Using device: {device}")

    # Environment setup
    env = gym.make('Walker2d-v5')
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Training parameters
    N_GAMES = 1000
    HORIZON = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    ALPHA = 0.0003
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = HORIZON
    num_steps = 1000000
    
    save_dir = create_save_directory()
    
    # Initialize agent with device
    agent = PPOAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        n_epochs=N_EPOCHS,
        horizon=HORIZON,
        device=device,
        policy_clip=0.2
    )
    
    #best_score = env.reward_range[0]
    best_score = float('-inf')
    score_history = []
    figure_file = os.path.join(save_dir, 'learning_curve.png')
    
    total_steps = 0
    episode = 0
    
    save_model(agent, save_dir, prefix='initial_')
    
    while total_steps < num_steps:
        observation, _ = env.reset()
        done = False
        truncated = False
        score = 0
        steps_in_episode = 0
        
        # Convert initial observation to tensor and move to device
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        
        while not (done or truncated):
            action, prob, val = agent.choose_action(observation)
            
            # Move action to CPU for environment step
            scaled_action = (action.cpu().numpy() if isinstance(action, torch.Tensor) else action) * max_action
            
            next_observation, reward, done, truncated, _ = env.step(scaled_action)
            
            # Convert next_observation to tensor and move to device
            next_observation = torch.tensor(next_observation, dtype=torch.float32).to(device)
            
            agent.store_transition(observation, action, prob, val, reward, done)
            
            observation = next_observation
            score += reward
            steps_in_episode += 1
            total_steps += 1
            
            if agent.is_ready_to_learn():
                agent.learn()
                
            if steps_in_episode >= 1000000:
                break

            if (total_steps) % PRINT_INTERVAL == 0:
                print(f'Episode: {episode + 1}, Steps: {steps_in_episode}, Total Steps: {total_steps}')
                print(f'Score: {score:.1f}, Avg Score: {avg_score:.1f}, Best Score: {best_score:.1f}')
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:]) if len(score_history) >= 100 else np.mean(score_history)
        
        if avg_score > best_score:
            best_score = avg_score
            save_model(agent, save_dir, is_best=True)
        
        if (episode + 1) % SAVE_INTERVAL == 0:
            #save_model(agent, save_dir, prefix=f'checkpoint_episode_{episode+1}_')
            plot_learning_curve(score_history, figure_file)
        
        
        episode += 1
    
    save_model(agent, save_dir, prefix='final_')
    plot_learning_curve(score_history, figure_file)
    np.save(os.path.join(save_dir, 'training_history.npy'), np.array(score_history))
    env.close()

if __name__ == '__main__':
    main()