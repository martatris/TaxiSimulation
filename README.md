
🚕 Reinforcement Learning Taxi Simulation (DQN)
===========================================================

-----------------------------------------------------------
OVERVIEW
-----------------------------------------------------------
This project implements a reinforcement learning agent that learns
how to operate a taxi service inside a grid world environment.

The taxi must:

 - Navigate across a city grid.
 - Pick up multiple passengers.
 - Drop them off at their destinations.
 - Optimize routes to maximize total reward.

The agent is trained using a Deep Q-Network (DQN), which combines
Q-learning with a neural network to approximate value functions.

-----------------------------------------------------------
FEATURES
-----------------------------------------------------------
- Custom Gymnasium environment (MultiPassengerTaxiEnv)
- Configurable grid size and number of passengers
- Reward shaping for pickups, drop-offs, and task completion
- DQN agent implemented with PyTorch
- Experience replay buffer for stable learning
- Adjustable episode length (max_steps)
- Training visualization (reward curve)
- Animation of the trained taxi behavior
- Libraries: Gymnasium, PyTorch, Matplotlib, NumPy

-----------------------------------------------------------
INSTALLATION
-----------------------------------------------------------
Make sure Python 3.8 or higher is installed.

Install required dependencies using:
> pip install gymnasium torch matplotlib numpy

-----------------------------------------------------------
HOW TO RUN
-----------------------------------------------------------

1. Run the training script:
   > python taxi_rl.py

This will:
 - Train the DQN model over multiple episodes.
 - Display live training progress in the console.
 - Plot the smoothed average reward curve at the end.

2. After training, an animation will show how the trained
   taxi moves around the grid, picking up and dropping off
   passengers efficiently.

-----------------------------------------------------------
CONFIGURATION OPTIONS
-----------------------------------------------------------
Inside taxi_rl.py, you can modify these parameters:

Parameter       | Description                               | Default
----------------|-------------------------------------------|----------
grid_size       | Width/height of the grid world            | 10
num_passengers  | Number of passengers per episode          | 2
episodes        | Number of training episodes               | 3000
max_steps       | Steps per episode (longer = more learning)| 1500
gamma           | Discount factor for future rewards        | 0.99
lr              | Learning rate for DQN optimizer           | 0.0005

To change episode duration, modify:
    rewards, model = train_dqn(env, episodes=3000, max_steps=1500)

-----------------------------------------------------------
REWARD SYSTEM
-----------------------------------------------------------
Action / Event             | Reward | Description
---------------------------|--------|------------------------------
Move (each step)           | -0.5   | Small time penalty
Successful pickup           | +15    | Taxi picks up a passenger
Successful drop-off         | +30    | Passenger dropped successfully
Wrong pickup/drop-off       | -10    | Incorrect action location
All passengers completed    | +100   | Mission success bonus

-----------------------------------------------------------
VISUAL OUTPUT
-----------------------------------------------------------
At the end of training:
 - A line chart shows average reward improvement.
 - An animation displays the taxi’s learned behavior.

Animation legend:
 - Red square = Taxi
 - Blue circle = Passenger waiting
 - Green box = Passenger destination

-----------------------------------------------------------
OPTIONAL (PORTFOLIO TIP)
-----------------------------------------------------------
To save the animation as a video or GIF:
Uncomment or add the following lines in the animation section:

    anim.save("taxi_agent_demo.mp4", writer="ffmpeg")

This helps showcase the project in your portfolio or GitHub.

-----------------------------------------------------------
FUTURE IMPROVEMENTS
-----------------------------------------------------------
- Add obstacles or traffic patterns to the grid.
- Allow dynamic passenger requests.
- Integrate Stable Baselines3 for comparison.
- Train multiple taxis (multi-agent system).

-----------------------------------------------------------
LEARNING OBJECTIVES
-----------------------------------------------------------
This project demonstrates:
 - Reinforcement learning fundamentals
 - Deep Q-learning with replay memory
 - Custom Gymnasium environment design
 - Reward shaping and convergence tracking
 - Visualization of intelligent agent behavior
