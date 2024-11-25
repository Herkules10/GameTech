import gymnasium as gym
import numpy as np
import torch
from ppo_agent import PPOAgent

def visualize_model():
    # Create environment with rendering
    env = gym.make('Walker2d-v5', render_mode='human')
    
    # Get environment dimensions
    input_dims = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Initialize agent
    agent = PPOAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        batch_size=64,
        alpha=0.0003,
        n_epochs=10,
        horizon=2048,
        device="cpu"
    )
    
    # Load the saved model
    actor_path = 'saved_models/best_actor.pth'
    critic_path = 'saved_models/best_critic.pth'
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    
    # Set to evaluation mode
    agent.actor.eval()
    agent.critic.eval()
    
    num_episodes = 1000
    for episode in range(num_episodes):
        observation, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        
        # Convert initial observation to tensor
        observation = torch.FloatTensor(observation)
        
        while not (done or truncated):
            # Get action from agent
            with torch.no_grad():
                action, _, _ = agent.choose_action(observation)
                
            # Scale and prepare action
            scaled_action = action.cpu().numpy() * max_action
            
            # Take action in environment
            observation, reward, done, truncated, _ = env.step(scaled_action)
            
            # Convert next observation to tensor
            observation = torch.FloatTensor(observation)
            
            total_reward += reward
        
        print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")
        
        if episode >= num_episodes - 1:
            break
    
    env.close()

if __name__ == "__main__":
    visualize_model()