
# Traffic Light Control System with Double DQN (LSTM) and SUMO

## Introduction

This project implements a **Traffic Light Control System** using **Double Deep Q-Network (DQN)** with **LSTM** to optimize traffic light timings in urban traffic environments. The project is powered by the **SUMO (Simulation of Urban Mobility)** platform to simulate traffic, and reinforcement learning is used to dynamically adjust the phases and durations of traffic lights.

The **Double DQN** algorithm ensures stable learning by decoupling the target and policy networks, while the **LSTM** layers help capture temporal dependencies in traffic flow data. The goal is to reduce traffic congestion, minimize waiting times, and improve traffic flow efficiency.

## YOLOv11x: Real-World Eyes of the Model

In real-world deployment, this system uses **YOLOv11x** for real-time vehicle detection. YOLOv11x is a state-of-the-art object detection model, fine-tuned for vehicle detection (e.g., cars, buses, motorbikes, trucks). It acts as the "eyes" of the model, helping to gather the real-time traffic data required by the reinforcement learning agent.

### How YOLOv11x is Used:
- **Vehicle Detection**: YOLOv11x detects vehicles and provides bounding boxes and classifications in real-time video feeds.
- **State Extraction**: Using YOLOv11x, a custom function calculates the **number of vehicles** and their **average waiting time** by tracking stationary and moving vehicles in a specified region.
- **Real-Time Feedback**: The extracted state information is fed into the Double DQN agent to optimize traffic light phases and durations.

This real-time vehicle detection and state extraction allow the system to effectively adapt to changing traffic patterns and minimize congestion.

## Features

- **Double DQN with LSTM**: Agents use a Double Deep Q-Network architecture combined with LSTM layers to capture traffic flow trends and predict the optimal action (traffic light phase and duration).
- **Traffic Light Control**: Traffic lights at multiple intersections are controlled dynamically by reinforcement learning agents to minimize traffic congestion.
- **YOLOv11x for Vehicle Detection**: YOLOv11x is used for real-time detection and tracking of vehicles in video streams, allowing the system to gather accurate traffic data.
- **Experience Replay**: The agents store and sample past experiences in a replay memory to stabilize learning.
- **SUMO Simulation**: Traffic simulation is done using the SUMO platform, a powerful tool for modeling real-world traffic scenarios.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/traffic-light-control-sumo.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd traffic-light-control-sumo
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install SUMO**:

   SUMO is required to simulate the traffic environment. Install it using the following commands:

   ```bash
   sudo add-apt-repository ppa:sumo/stable
   sudo apt-get update
   sudo apt-get install sumo sumo-tools sumo-doc
   ```

5. **Install YOLOv11x (Ultralytics)**:

   ```bash
   pip install ultralytics
   ```

6. **Mount Google Drive (for Colab)**:

   If you're using Google Colab, make sure to mount Google Drive to access video datasets and save trained models:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

## Usage

### Simulation

1. **Set the configuration path and base path** in `simulation.py`. These paths should point to your SUMO configuration file and base project directory, respectively:

   ```python
   if __name__ == "__main__":
       config_path = r"configuration.sumocfg"  # Path to the SUMO configuration file
       base_path = r"/content/drive/MyDrive/Colab Notebooks"  # Base path for the project
       simulation = TrafficSimulation(
           config_path, 
           base_path, 
           epochs=100, 
           steps=1000, 
           window_size=32, 
           train=True
       )
       simulation.run_simulation()
   ```

2. **Run the traffic light control simulation**:

   ```bash
   python simulation.py
   ```

3. **Run YOLOv11x vehicle detection and state extraction**:

   YOLOv11x is used to detect vehicles and calculate their waiting times in real-time video streams. Run the detection process using:

   ```python
   python yolo_vehicle_detection.py
   ```

   This script processes the video and outputs the detected vehicles and their corresponding waiting times. It integrates seamlessly with the reinforcement learning model to provide accurate traffic data for decision-making.

### Results

- **Model checkpoints**: The model weights will be saved in the `models` directory periodically.
- **Plots**: After training, the script generates plots that display the average waiting time and cumulative rewards across epochs. These plots are saved in the `plots` directory.

## Project Structure

- **agents.py**: Contains the `BaseAgent` class, which defines the reinforcement learning agents using Double DQN with LSTM.
- **memory.py**: Implements experience replay with the `ReplayMemory` class, used to store and sample past experiences.
- **models.py**: Defines the LSTM-based neural network model used by each agent to predict actions.
- **simulation.py**: The main script for running the SUMO traffic simulation and training the agents.
- **yolo_vehicle_detection.py**: Script for detecting vehicles and calculating average waiting times using YOLOv11x.
- **requirements.txt**: A file listing all the Python dependencies required to run the project.
- **README.md**: This documentation file.

## Dependencies

- **Python** (>= 3.7)
- **SUMO**: Traffic simulation platform.
- **TensorFlow** (>= 2.0): Used for building and training the neural networks.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting the results.
- **TraCI**: SUMO's Traffic Control Interface, used for controlling the traffic lights from Python.
- **sumolib**: Python library for working with SUMO.
- **YOLOv11x (Ultralytics)**: For real-time vehicle detection in videos.
- **OpenCV**: For video frame processing and bounding box drawing.

## Results

- **Average Waiting Time Across Epochs**: This plot shows how the average waiting time decreases over time as the agents learn better traffic control strategies.
- **Cumulative Reward Across Epochs**: This plot shows the cumulative rewards collected by the agents during training. Higher rewards indicate better performance in controlling traffic.

Both of these results are saved as images in the `plots` directory after the simulation is run.
