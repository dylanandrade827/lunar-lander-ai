import gymnasium as gym

def test_env():
    env = gym.make("LunarLander-v3", render_mode=None)
    obs, info = env.reset()
    print("Observation shape:", obs.shape)
    done = False
    total_reward = 0.0

    while not done:
        action = env.action_space.sample()  # random
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print("Random policy episode return:", total_reward)
    env.close()

if __name__ == "__main__":
    test_env()
