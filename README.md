
# Traffic Light Management System with Multi-Agent Reinforcement Learning

This project implements a traffic light control system using Multi-Agent Reinforcement Learning (MARL) in SUMO (Simulation of Urban Mobility). The system is designed to optimize traffic flow across multiple junctions by selecting the appropriate lane and phase duration for traffic lights.

## Project Structure

- `TrafficModel`: A class that constructs and trains a neural network to predict the best actions (lane selection and phase duration) based on the traffic state.
- `ReplayMemory`: A class to store past experiences (state, action, reward, new state) for training the model using experience replay.
- `BaseAgent`: A class implementing the reinforcement learning agent, which selects actions based on the current traffic state and updates the model with the learning process.
- `TrafficSimulation`: A class that interacts with the SUMO simulator, gathers traffic data, and controls traffic lights using the trained agents.

## Dependencies

The following dependencies are required to run the project:

- Python 3.x
- SUMO (Simulation of Urban Mobility)
- TensorFlow
- Keras
- Numpy
- Matplotlib
- Traci (SUMO Python API)
- Sumolib (SUMO Python libraries)

You can install these dependencies using the following commands:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

pip install pyserial
pip install sumolib traci tensorflow keras numpy matplotlib
```

## How the System Works

1. **Model Architecture**: 
   The system uses a neural network with LSTM layers to handle the sequential nature of traffic data. The network takes a series of traffic states as input and predicts two outputs:
   - Lane selection for the traffic light.
   - Phase duration for the selected lane.
   
2. **Agents**: 
   Each traffic junction is controlled by an agent that selects actions (lane and phase duration) to optimize traffic flow. The agent uses a reinforcement learning algorithm with epsilon-greedy exploration to balance exploration and exploitation.

3. **Simulation**: 
   The SUMO simulator is used to model the traffic environment. Traffic data is collected, and the agents interact with the environment to make decisions about traffic light control.

4. **Training**:
   The agents are trained over multiple epochs using the traffic data and feedback from the environment. The reward is calculated based on the reduction in waiting time at the junctions.

5. **Evaluation**: 
   After training, the system can be evaluated by running the simulation and observing the traffic flow and average waiting time at each junction.

## Files and Directories

- `models/`: Contains saved model weights for each traffic junction.
- `plots/`: Contains plots of average waiting time and cumulative reward across epochs.

## Running the Simulation

To run the simulation, simply execute the following command:

```bash
python <script_name>.py
```

Ensure that SUMO is installed and configured correctly before running the simulation. Modify the `config_path` and `base_path` variables in the script to point to your SUMO configuration file and project directory, respectively.

## Results

- The system generates plots of average waiting time and cumulative rewards across the epochs. These plots help visualize the performance of the traffic control system.

## Author

Hassan

## License

This project is licensed under the MIT License.
