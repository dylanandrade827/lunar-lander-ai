import gymnasium as gym
import torch
import yaml
from pathlib import Path

from agents.dqn_agent import DQNAgent
from utils.logger import CSVLogger
from utils.reward_shaping import shaped_reward
from utils.curriculum_env_wrapper import CurriculumLunarLander

def train(config_path, log_dir="experiments/baseline_run"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = gym.make(config["env_name"])
    state, info = env.reset(seed=config["seed"])

    state_dim = len(state)
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DQNAgent(state_dim, action_dim, config, device=device)
    logger = CSVLogger(log_dir)

    for episode in range(config["num_episodes"]):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_loss = None

        while not done and steps < config["max_steps_per_episode"]:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if config.get("use_reward_shaping", False):
                reward = shaped_reward(next_state, reward, done)

            if config.get("use_curriculum, False"):
                env = CurriculumLunarLander(env, easy_episodes=config.get("easy_episodes, 200"))

            agent.push_transition(state, action, reward, next_state, done)

            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            agent.total_steps += 1
            state = next_state
            total_reward += reward
            steps += 1

        agent.maybe_update_epsilon(episode)
        logger.log(episode, total_reward, agent.epsilon, steps, last_loss if last_loss else 0.0)
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    logger.close()
    env.close()

if __name__ == "__main__":
    Path("experiments/baseline_run").mkdir(parents=True, exist_ok=True)
    train("experiments/config_baseline.yaml", log_dir="experiments/baseline_run")
