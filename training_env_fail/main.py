import gym_env
import td3_agent
from replay_buffer import ReplayBuffer
import torch
from matplotlib import pyplot as plt

cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

# Name of file to which avg steps will be saved as well as best episode steps
file_path = "avg_steps.txt"

# Name used to load and save agent's weights and biases
# ie. if set to test it will be like "test_agent"
# same for critic and actor and critic optimizers
to_load = "test"
to_save = "test"

# Should be set to "human" to be able to see the agent play
# Anything else will make it not visible
mode = "humn"

# Read the file and get the last line
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()
except:
    lines = 0

# if the program was run for >=10 episodes, it will try to load the last index
if lines:
    last_line = lines[-1].strip()  # Remove newline characters and leading/trailing whitespaces
    index, _, _ = last_line.split('\t')
    index = int(index)

    random_steps = 0
    print(f"Last index: {index}")
else:
    index = 0
    print("The file is empty.")
    random_steps = 1000

# load the env
env = gym_env.BulletHell(mode)

# Number of episodes and epochs to run
num_episodes = 10
num_epochs = 300
total_steps = 0

batch_size = 256

max_action = env.action_space.high[0]
action_shape = env.action_space.shape

input_shape = env.observation_space.shape

# Input shape is set to 42*5 becouse each bullet has 5 values = position(2), velozity(2), size(1)
# and the env is set to have a maximum of 40 bullets
# + 5 values for the agent [position(2), velozity(2), size(1)]
agent = td3_agent.TD3(42*5, 5, torch.tensor(max_action).to(device))

#agent = DDPG(42*5, 5, torch.tensor(max_action).to(device))

# load agents weights and biases
try:
    agent.load(to_load)
except:
    print("Failed to load agent")

training_avg = 0
training_avg_list = []

# create replay buffer by setting input size and action space
replay_buffer = ReplayBuffer(42*5, 5)

epoch_nr = 1

for epoch in range(num_epochs):
    total_steps = 0
    best_episode = 0
    for episode in range(num_episodes):
        # Reset the environment for each episode
        state = env.reset()

        # Count steps
        steps = 0
        
        # Initialize variables for the episode
        total_reward = 0
        done = False
        
        while not done:
            # Render the environment (optional)
            env.render()
            
            # agent actions are treated as propabilities of moving in any of the 4 cardinal directions
            # or not moving, the highest percentage is selected as the action that agent takes
            # but before it is selected, some noise is added to make the agent explore more
            action = agent.select_action(torch.tensor(state, dtype=torch.float32).to(device))
            action  = torch.tensor(action, dtype=torch.float32).to(device)
            noise = (
                torch.randn_like(action) * 0.2
            ).clamp(-0.5, 0.5)
            action = action + noise
            a = action.argmax()

            # Take a step in the environment
            next_state, reward, done, _ = env.step(a)

            # Store in Replay Buffer
            replay_buffer.add(torch.tensor(state, dtype=torch.float32).cpu().flatten(), 
                              action.cpu(), 
                              torch.tensor(next_state, dtype=torch.float32).cpu().flatten(), reward, done)

            if replay_buffer.size > batch_size:
                agent.train(replay_buffer, batch_size)

            # Accumulate the total reward for the episode
            total_reward += reward

            state = next_state

            # Count total steps for session, steps in episode
            total_steps += 1
            steps += 1
            if steps > best_episode: best_episode = steps
        
        total_steps += steps

    # Collect and print avg steps for each epoch
    avg = total_steps/num_episodes
    print(f"[{epoch_nr}]\tSurvived an avarage of {round(avg,2)} steps. Best episode lasted {best_episode}")
    training_avg += avg

    # Write each epoch avg steps to the file
    with open(file_path, 'a') as file:
        index += num_episodes
        file.write(f"{index}\t{avg}\t{best_episode}\n")
    training_avg_list.append(avg)
    epoch_nr += 1
    # Save the agent after each epoch
    agent.save(to_save)

# print this session avg steps
training_avg = training_avg/num_epochs
print(f"This session avg steps: {training_avg}")

# Close the environment after testing and save the agent
env.close()
agent.save(to_save)

# Draw a plot for the current session
plt.plot(training_avg_list)
plt.ylabel("steps survived")
plt.xlabel("Epoch")
plt.show()