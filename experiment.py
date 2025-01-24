import argparse
import json
import os.path as osp
import pathlib
import pickle
import time
import numpy as np
import tensorflow as tf
import sys

from env.deepmimic_env import DeepMimicEnv
from learning.rl_world import RLWorld
from util.arg_parser import ArgParser
import util.mpi_util as MPIUtil

class AdaptiveRewardWeighting:
    def __init__(self, window_size=100, alpha=0.1):
        self.window_size = window_size
        self.alpha = alpha
        self.task_losses = []
        self.style_losses = []
        self.current_weight = 0.5  # Start with equal weighting

    def update(self, task_loss, style_loss):
        self.task_losses.append(task_loss)
        self.style_losses.append(style_loss)
        
        if len(self.task_losses) > self.window_size:
            self.task_losses.pop(0)
            self.style_losses.pop(0)

        if len(self.task_losses) >= self.window_size:
            avg_task = np.mean(self.task_losses)
            avg_style = np.mean(self.style_losses)
            ratio = avg_task / (avg_style + 1e-8)
            
            # Update weight using exponential moving average
            target_weight = 1.0 / (1.0 + ratio)
            self.current_weight = (1 - self.alpha) * self.current_weight + self.alpha * target_weight

        return self.current_weight

def run_experiment(args, out_dir):
    # Setup
    arg_parser = ArgParser()
    arg_parser.load_args(args)
    
    env = DeepMimicEnv(args, enable_draw=False)
    world = RLWorld(env, arg_parser)
    agent = world.agents[0]
    
    # Print diagnostics
    print("\nInitial Configuration:")
    print(f"Environment: {env.get_name()}")
    print(f"Agent type: {type(agent).__name__}")
    print(f"State size: {env.get_state_size(agent.id)}")
    print(f"Action size: {env.get_action_size(agent.id)}")
    print(f"Initial samples needed: {agent.init_samples}")
    print(f"Replay buffer size: {agent.replay_buffer_size}")
    
    # Important: Initialize agent's training path
    agent.path.clear()
    
    num_steps = int(arg_parser.parse_int('num_steps', 2e7))
    eval_interval = 50
    
    episode_rewards = []
    episode = 0
    total_steps = 0
    
    print("\nStarting training loop...")
    
    while total_steps < num_steps:
        # Clear old paths
        agent.path.clear()
        world.reset()
        
        episode_steps = 0
        episode_reward = 0
        
        # Track states and rewards for debugging
        states = []
        actions = []
        rewards = []
        
        while not env.is_episode_end() and episode_steps < 1000:
            # Update world and agents
            world.update(1.0/60.0)
            
            # Record data if agent needs new action
            if agent.need_new_action():
                # Get state and store it
                state = env.record_state(agent.id)
                states.append(state)
                
                # Store goal if used 
                goal = env.record_goal(agent.id) if env.get_goal_size(agent.id) > 0 else None
                
                # Get action and store it
                action, logp = agent._decide_action(state, goal)
                actions.append(action)
                
                # Apply action
                env.set_action(agent.id, action)
                
                # Calculate rewards
                reward = env.calc_reward(agent.id)
                rewards.append(reward)
                
                # Store transition in path
                agent.path.states.append(state)
                if goal is not None:
                    agent.path.goals.append(goal)
                agent.path.actions.append(action)
                agent.path.logps.append(logp)
                agent.path.rewards.append(reward)
                
                # Update total reward
                episode_reward += reward
                
            # Check if we should train
            if agent.path.get_pathlen() > 0:
                world.enable_training = True
                if agent.enable_training:
                    agent._update_new_action()
            
            episode_steps += 1
            total_steps += 1
        
        # End episode
        if agent.path.get_pathlen() > 0:
            agent.end_episode()
            episode_rewards.append(episode_reward)
        
        # Print debug info
        if episode % eval_interval == 0:
            print(f"\nEpisode {episode} Debug Info:")
            print(f"Total Steps: {total_steps}")
            print(f"Path Length: {agent.path.get_pathlen()}")
            print(f"Episode Steps: {episode_steps}")
            print(f"Number of States: {len(states)}")
            print(f"Number of Actions: {len(actions)}")
            print(f"Number of Rewards: {len(rewards)}")
            if rewards:
                print(f"Reward Stats:")
                print(f"  Min: {min(rewards):.3f}")
                print(f"  Max: {max(rewards):.3f}")
                print(f"  Mean: {np.mean(rewards):.3f}")
            print(f"Episode Reward: {episode_reward:.3f}")
            print(f"Replay Buffer Size: {agent.replay_buffer.get_current_size()}")
            print(f"Training Enabled: {world.enable_training}")
            print(f"Agent Training Enabled: {agent.enable_training}")
            
            # Print some state/action samples
            if states:
                print(f"\nFirst State Sample: {states[0][:5]}...")
            if actions:
                print(f"First Action Sample: {actions[0]}") 
                
            # Every 500 episodes save checkpoint
            if episode % 500 == 0 and agent.enable_training:
                save_path = osp.join(out_dir, f"model_{episode}.ckpt")
                agent.save_model(save_path)
        
        episode += 1

    # Save final results...

    # Rest of the saving code...

    # Save final results
    final_results = {
        "walk": {
            "rewards": episode_rewards,
            "task_losses": task_losses,
            "style_losses": style_losses,
        }
    }
    
    final_info = {
        "walk": {
            "means": {
                "final_reward": np.mean(episode_rewards[-eval_interval:]),
                "final_task_loss": np.mean(task_losses[-eval_interval:]),
                "final_style_loss": np.mean(style_losses[-eval_interval:]),
                "training_time": episode_time,
                "total_steps": total_steps,
                "total_episodes": episode
            }
        }
    }

    # Save results
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    with open(osp.join(out_dir, "final_info.json"), "w") as f:
        json.dump(final_info, f)
        
    with open(osp.join(out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(final_results, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_file", type=str, required=True, help="Path to DeepMimic args file")
    parser.add_argument("--out_dir", type=str, default="run_0", help="Output directory") 
    args = parser.parse_args()

    # Convert args file into format expected by DeepMimic
    with open(args.arg_file, 'r') as f:
        arg_str = f.read().splitlines()
    
    # Split each line into separate arguments
    processed_args = []
    for line in arg_str:
        line = line.strip()
        if line and not line.startswith('#'):  # Skip empty lines and comments
            processed_args.extend(line.split())
    
    print("PROCESSED ARGS:", processed_args)
    run_experiment(args=processed_args, out_dir=args.out_dir)