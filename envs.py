# import gym

# def run_environment(env_name, episodes=20, render=True):
#     env = gym.make(env_name, render_mode="human")
    
#     for episode in range(episodes):
#         observation, _ = env.reset()  # New Gym versions return (obs, info)
#         done = False
#         total_reward = 0
        
#         while not done:
#             if render:
#                 env.render()
            
#             action = env.action_space.sample()
#             observation, reward, done, truncated, info = env.step(action)
#             total_reward += reward
            
#             if truncated:
#                 done = True
        
#         print(f"Episode {episode + 1}: Total Reward = {total_reward}")
    
#     env.close()

# if __name__ == "__main__":
#     print("Running CartPole-v1")
#     run_environment("CartPole-v1")
    
#     # print("Running MountainCar-v0")
#     # run_environment("MountainCar-v0")

import sqlite3
conn = sqlite3.connect('experiments.db')
conn.execute("DELETE FROM episodes WHERE agent='v4_novelty_seeker'")
conn.execute("DELETE FROM hypotheses WHERE agent='v4_novelty_seeker'")
conn.execute("DELETE FROM novelty_scores WHERE agent='v4_novelty_seeker'")
conn.execute("DELETE FROM experiments WHERE agent='v4_novelty_seeker'")
conn.commit()
conn.close()
print('V4 data cleared.')