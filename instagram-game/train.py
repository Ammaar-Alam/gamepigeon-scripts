import numpy as np
import torch
from ppo_agent import PPOAgent
from ppo_game_env import GameEnvironment
import time
from collections import defaultdict
import signal
import sys
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\nTraining interrupted. Cleaning up...")
    if 'env' in globals():
        env.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def compute_gae(rewards, values, dones, next_value, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    advantages = []
    advantage = 0
    
    for reward, value, done, next_val in zip(
        rewards[::-1], values[::-1], dones[::-1],
        [next_value] + values[:-1][::-1]
    ):
        if done:
            delta = reward - value
            advantage = delta
        else:
            delta = reward + gamma * next_val - value
            advantage = delta + gamma * lam * advantage
            
        advantages.append(advantage)
        
    advantages = advantages[::-1]
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    return advantages, returns

def collect_rollouts(env, agent, num_steps=2048):
    """Collect a batch of simulated experiences."""
    rollouts = defaultdict(list)
    observation, _ = env.reset()  # Initial reset
    
    episode_rewards = []
    current_episode_reward = 0
    episode_length = 0
    num_episodes = 0
    total_hits = 0
    
    steps_taken = 0
    while steps_taken < num_steps:
        try:
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = agent.actor_critic.get_action(observation)
                
            # Execute action in environment
            next_observation, reward, done, truncated, _ = env.step(action.cpu().numpy())
            
            # Store transition
            rollouts['screens'].append(observation['screen'])
            rollouts['ball_positions'].append(observation['ball_pos'])
            rollouts['ball_velocities'].append(observation['ball_vel'])
            rollouts['actions'].append(action.cpu().numpy())
            rollouts['log_probs'].append(log_prob.cpu().numpy())
            rollouts['rewards'].append(reward)
            rollouts['values'].append(value.cpu().numpy())
            rollouts['dones'].append(done)
            
            current_episode_reward += reward
            episode_length += 1
            steps_taken += 1
            
            if done:
                total_hits += env.hits  # Get hits from environment
                print(f"\n[Episode {num_episodes + 1}]")
                print(f"Length: {episode_length}")
                print(f"Reward: {current_episode_reward:.2f}")
                print(f"Hits: {env.hits}")
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                episode_length = 0
                num_episodes += 1
                observation, _ = env.reset()
            else:
                observation = next_observation
                
        except Exception as e:
            print(f"Error during rollout collection: {str(e)}")
            observation, _ = env.reset()
            continue
    
    # Get final value for GAE
    with torch.no_grad():
        _, _, next_value = agent.actor_critic.get_action(observation)
        next_value = next_value.cpu().numpy()
    
    # Calculate advantages and returns
    advantages, returns = compute_gae(
        rollouts['rewards'],
        rollouts['values'],
        rollouts['dones'],
        next_value
    )
    
    rollouts['advantages'] = advantages
    rollouts['returns'] = returns
    
    # Convert to numpy arrays
    for k, v in rollouts.items():
        rollouts[k] = np.array(v)
    
    return rollouts, episode_rewards, num_episodes, total_hits

def train():
    """Main training loop."""
    try:
        # Create environment and agent
        env = GameEnvironment()
        agent = PPOAgent(
            state_dim=(84, 84, 1),  # Screen dimensions
            action_dim=1,  # Bar position
            lr=3e-4,
            gamma=0.99,
            epsilon=0.2
        )
        
        # Training parameters
        num_epochs = 1000
        steps_per_epoch = 2048
        save_freq = 5  # Save more frequently
        
        # Training metrics
        best_mean_reward = float('-inf')
        best_total_hits = 0
        episode_rewards_history = []
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")
            start_time = time.time()
            
            try:
                # Collect rollouts
                rollouts, episode_rewards, num_episodes, total_hits = collect_rollouts(env, agent, steps_per_epoch)
                episode_rewards_history.extend(episode_rewards)
                
                # Update policy
                policy_loss, value_loss, entropy = agent.update(rollouts)
                
                # Calculate statistics
                mean_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                
                # Print training statistics
                print(f"\nTraining Stats:")
                print(f"Episodes completed: {num_episodes}")
                print(f"Total hits: {total_hits}")
                print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                print(f"Policy loss: {policy_loss:.4f}")
                print(f"Value loss: {value_loss:.4f}")
                print(f"Entropy: {entropy:.4f}")
                print(f"Time elapsed: {time.time() - start_time:.2f}s")
                
                # Save best model based on both reward and hits
                save_model = False
                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    save_model = True
                    print(f"New best reward: {best_mean_reward:.2f}")
                
                if total_hits > best_total_hits:
                    best_total_hits = total_hits
                    save_model = True
                    print(f"New best hits: {best_total_hits}")
                
                if save_model:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"models/ppo_model_best_{timestamp}.pt"
                    agent.save(model_path)
                    print(f"New best model saved: {model_path}")
                
                # Save periodic checkpoint
                if (epoch + 1) % save_freq == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_path = f"models/ppo_model_epoch_{epoch+1}_{timestamp}.pt"
                    agent.save(model_path)
                    print(f"Checkpoint saved: {model_path}")
                    
            except Exception as e:
                print(f"Error during epoch: {str(e)}")
                continue
                
            # Save after each epoch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/ppo_model_epoch_{epoch+1}_{timestamp}.pt"
            agent.save(model_path)
            print(f"Checkpoint saved: {model_path}")
                
    except Exception as e:
        print(f"Training error: {str(e)}")
    finally:
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/ppo_model_final_{timestamp}.pt"
        agent.save(model_path)
        print(f"Final model saved: {model_path}")
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    print("Starting PPO training for Instagram game...")
    print("Make sure iPhone Mirroring is running and the game is ready")
    
    try:
        train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure environment is closed properly
        if 'env' in globals():
            env.close()
