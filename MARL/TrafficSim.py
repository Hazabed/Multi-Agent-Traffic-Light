import numpy as np
import os
import matplotlib.pyplot as plt
import traci
from sumolib import checkBinary
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# TrafficModel class handles neural network construction and training
class TrafficModel:
    def __init__(self, input_dims, n_actions_1, n_actions_2, lr, window_size=4):
        self.input_dims = input_dims
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_2
        self.lr = lr
        self.window_size = window_size
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=(self.window_size, self.input_dims))
        x = LSTM(128, return_sequences=True)(inputs)  # Use default activation functions
        x = LSTM(64)(x)
        x = Dense(32, activation='relu')(x)

        output_1 = Dense(self.n_actions_1, activation='linear', name='a_lane_output')(x)
        output_2 = Dense(self.n_actions_2, activation='linear', name='b_duration_output')(x)

        model = Model(inputs=inputs, outputs=[output_1, output_2])
        model.compile(optimizer=Adam(learning_rate=self.lr),
                      loss={'a_lane_output': MeanSquaredError(), 'b_duration_output': MeanSquaredError()})
        return model

    def predict(self, state_sequence):
        return self.model.predict(state_sequence, verbose=0)

    def fit(self, state_batch, q_targets, batch_size):
        self.model.fit(state_batch, q_targets, verbose=0, batch_size=batch_size)

    def update_weights(self, model):
        self.model.set_weights(model.get_weights())

class ReplayMemory:
    def __init__(self, max_size, input_dims, window_size=4):
        self.max_size = max_size
        self.memory_counter = 0
        self.window_size = window_size
        self.state_memory = np.zeros((max_size, window_size, input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((max_size, window_size, input_dims), dtype=np.float32)
        self.action_memory1 = np.zeros(max_size, dtype=np.int32)
        self.action_memory2 = np.zeros(max_size, dtype=np.int32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)

    def store_transition(self, state_sequence, action1, action2, reward, new_state_sequence, done):
        index = self.memory_counter % self.max_size
        self.state_memory[index] = state_sequence
        self.new_state_memory[index] = new_state_sequence
        self.action_memory1[index] = action1
        self.action_memory2[index] = action2
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.max_size)
        if max_mem < batch_size:
            return None  # Not enough data
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (self.state_memory[batch],
                self.new_state_memory[batch],
                self.action_memory1[batch],
                self.action_memory2[batch],
                self.reward_memory[batch],
                self.terminal_memory[batch])

class BaseAgent:
    def __init__(self, gamma, epsilon, lr, input_dims, n_actions_1, n_actions_2, batch_size, memory, window_size=4, epsilon_min=0.01, target_update_interval=100):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = None  # Will be set dynamically
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.learn_step_counter = 0
        self.n_actions_1 = n_actions_1
        self.n_actions_2 = n_actions_2
        self.window_size = window_size

        self.memory = memory

        self.q_eval = TrafficModel(input_dims, n_actions_1, n_actions_2, lr, window_size=window_size)
        self.q_target = TrafficModel(input_dims, n_actions_1, n_actions_2, lr, window_size=window_size)
        self.q_target.update_weights(self.q_eval.model)

    def choose_action(self, state_sequence, is_train=True):
        state_input = np.expand_dims(state_sequence, axis=0)
        if is_train and np.random.random() > self.epsilon:
            actions = self.q_eval.predict(state_input)
            lane_q_values = actions[0]
            duration_q_values = actions[1]
            lane_action = np.argmax(lane_q_values[0])
            duration_action = np.argmax(duration_q_values[0])
            return lane_action, duration_action, False
        elif is_train:
            lane_action = np.random.choice(self.n_actions_1)
            duration_action = np.random.choice(self.n_actions_2)
            return lane_action, duration_action, True
        else:
            actions = self.q_eval.predict(state_input)
            lane_q_values = actions[0]
            duration_q_values = actions[1]
            lane_action = np.argmax(lane_q_values[0])
            duration_action = np.argmax(duration_q_values[0])
            return lane_action, duration_action, False

    def decay_epsilon(self):
        if self.epsilon_dec is not None:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_dec)

    def update_target_network(self):
        self.q_target.update_weights(self.q_eval.model)

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        result = self.memory.sample_buffer(self.batch_size)
        if result is None:
            return

        state_batch, new_state_batch, action_batch1, action_batch2, reward_batch, terminal_batch = result

        q_eval_lane, q_eval_duration = self.q_eval.predict(state_batch)
        q_next_lane, q_next_duration = self.q_target.predict(new_state_batch)

        q_target_lane = q_eval_lane.copy()
        q_target_duration = q_eval_duration.copy()

        for idx in range(self.batch_size):
            if terminal_batch[idx]:
                q_target_lane[idx, action_batch1[idx]] = reward_batch[idx]
                q_target_duration[idx, action_batch2[idx]] = reward_batch[idx]
            else:
                q_target_lane[idx, action_batch1[idx]] = reward_batch[idx] + self.gamma * np.max(q_next_lane[idx])
                q_target_duration[idx, action_batch2[idx]] = reward_batch[idx] + self.gamma * np.max(q_next_duration[idx])
                
        self.q_eval.fit(
            state_batch,
            {'a_lane_output': q_target_lane, 'b_duration_output': q_target_duration},
            batch_size=self.batch_size
        )

        self.decay_epsilon()
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_interval == 0:
            self.update_target_network()

class TrafficSimulation:
    def __init__(self, config_path, base_path, epochs, steps, model_name='model.weights', train=True, window_size=4):
        self.config_path = config_path
        self.base_path = base_path
        self.epochs = epochs
        self.steps = steps
        self.model_name = model_name
        self.train = train
        self.agents = {}
        self.window_size = window_size
        self.select_lane = {
            "clusterJ10_J2_J3_J4_#4more": [
                ["GGGGrrrrrrrrrrrrrrrrr", "yyyyrrrrrrrrrrrrrrrrr"],
                ["rrrrGGGGGGrrrrrrrrrrr", "rrrryyyyyyrrrrrrrrrrr"],
                ["rrrrrrrrrrGGGGGrrrrrr", "rrrrrrrrrryyyyyrrrrrr"],
                ["rrrrrrrrrrrrrrrGGGGGG", "rrrrrrrrrrrrrrryyyyyy"]
            ],
            "cluster11557834599_4394798929_9597262032_J2": [
                ["GGGGGGrrrrrGrrrrrrGrrrrrrr", "yyyyyyrrrrrGrrrrrrGrrrrrrr"],
                ["rrrrrrGGGGGGrrrrrrGrrrrrrr", "rrrrrryyyyyGrrrrrrGrrrrrrr"],
                ["rrrrrrrrrrrGGGGGGGGrrrrrrr", "rrrrrrrrrrrGyyyyyyGrrrrrrr"],
                ["rrrrrrrrrrrGrrrrrrGGGGGGGG", "rrrrrrrrrrrGrrrrrrGyyyyyyy"]
            ],
            "clusterJ57_J58_J59_J60_#4more": [
                ["GrrrrrrGrrrrrrrGrrrrrrGGGGGGGG", "GrrrrrrGrrrrrrrGrrrrrrGyyyyyyy"],
                ["GGGGGGGGrrrrrrrGrrrrrrGrrrrrrr", "GyyyyyyyrrrrrrrGrrrrrrGrrrrrrr"],
                ["GrrrrrrGGGGGGGGGrrrrrrGrrrrrrr", "GrrrrrrGyyyyyyyyrrrrrrGrrrrrrr"],
                ["GrrrrrrGrrrrrrrGGGGGGGGrrrrrrr", "GrrrrrrGrrrrrrrGyyyyyyyrrrrrrr"]
            ],
            "cluster11552825949_12066181616_12066181631_4394798946_#4more": [
                ["GGGGGGGGGrrrrrrrrGrrrrrrrGrrrrrrrr", "GyyyyyyyyrrrrrrrrGrrrrrrrGrrrrrrrr"],
                ["GrrrrrrrGGGGGGGGGGrrrrrrrGrrrrrrrr", "GrrrrrrrGyyyyyyyyyrrrrrrrGrrrrrrrr"],
                ["GrrrrrrrGrrrrrrrrGGGGGGGGGrrrrrrrr", "GrrrrrrrGrrrrrrrrGyyyyyyyyrrrrrrrr"],
                ["GrrrrrrrGrrrrrrrrGrrrrrrrGGGGGGGGG", "GrrrrrrrGrrrrrrrrGrrrrrrrGyyyyyyyy"]
            ],
            "cluster12065927671_12065927678_12065927679_12065927687_#4more": [
                ["GGGGGGGrrrrrrrrrrrrrrrrrrrrr", "Gyyyyyyrrrrrrrrrrrrrrrrrrrrr"],
                ["GrrrrrrGGGGGGGGrrrrrrGrrrrrr", "GrrrrrrGyyyyyyyrrrrrrGrrrrrr"],
                ["GrrrrrrGrrrrrrGGGGGGGGrrrrrr", "GrrrrrrGrrrrrrGyyyyyyyrrrrrr"],
                ["GrrrrrrGrrrrrrGrrrrrrGGGGGGG", "GrrrrrrGrrrrrrGrrrrrrGyyyyyy"]
            ],
            "clusterJ112_J113_J114_J115_#4more": [
                ["GGGGGGrrrrrrrrrrrrrrrrrrr", "yyyyyyrrrrrrrrrrrrrrrrrrr"],
                ["rrrrrrGGGGrrrrrrrrrrrrrrr", "rrrrrryyyyrrrrrrrrrrrrrrr"],
                ["rrrrrrrrrrGGGGGGGGrrrrrrr", "rrrrrrrrrryyyyyyyyrrrrrrr"],
                ["rrrrrrrrrrrrrrrrrrGGGGGGG", "rrrrrrrrrrrrrrrrrryyyyyyy"]
            ],
            "clusterJ42_J43_J44_J45_#4more": [
                ["GGGGGGGGrrrrrrrrrrrrrrrrr", "yyyyyyyyrrrrrrrrrrrrrrrrr"],
                ["rrrrrrrrGGGGGrrrrrrrrrrrr", "rrrrrrrryyyyyrrrrrrrrrrrr"],
                ["rrrrrrrrrrrrrGGGGGGrrrrrr", "rrrrrrrrrrrrryyyyyyrrrrrr"],
                ["rrrrrrrrrrrrrrrrrrrGGGGGG", "rrrrrrrrrrrrrrrrrrryyyyyy"]
            ],
            "cluster255538828_471024088_471024089_471024186": [
                ["GGGGGGGGGGGGrrrrrrrrrrrrrrrrrrrrrrrrrrrrr", "yyyyyyyyyyyyrrrrrrrrrrrrrrrrrrrrrrrrrrrrr"],
                ["rrrrrrrrrrrrGGGGGGGGGrrrrrrrrrrrrrrrrrrrr", "rrrrrrrrrrrryyyyyyyyyrrrrrrrrrrrrrrrrrrrr"],
                ["rrrrrrrrrrrrrrrrrrrrrGGGGGGGGGGrrrrrrrrrr", "rrrrrrrrrrrrrrrrrrrrryyyyyyyyyyrrrrrrrrrr"],
                ["rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrGGGGGGGGGG", "rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrryyyyyyyyyy"]
            ]
        }
    # Add the maximums dictionary
        self.maximums = {
            'cluster11552825949_12066181616_12066181631_4394798946_#4more': np.array([35, 50, 345, 55]),
            'cluster11557834599_4394798929_9597262032_J2': np.array([15, 25, 220, 75]),
            'cluster12065927671_12065927678_12065927679_12065927687_#4more': np.array([50, 20, 55, 45]),
            'cluster255538828_471024088_471024089_471024186': np.array([40, 25, 105, 215]),
            'clusterJ10_J2_J3_J4_#4more': np.array([65, 10, 45, 50]),
            'clusterJ112_J113_J114_J115_#4more': np.array([70, 65, 20, 25]),
            'clusterJ42_J43_J44_J45_#4more': np.array([70, 40, 75, 50]),
            'clusterJ57_J58_J59_J60_#4more': np.array([55, 55, 45, 30])
        }
    def initialize_agents(self, all_junctions):
        total_training_steps = self.epochs * (self.steps // self.window_size)
        epsilon_decay = (0.3 - 0.01) / total_training_steps  # Adjust epsilon decay

        for junction in all_junctions:
            input_dims = 12  # Adjusted to match your state size
            n_actions_1 = len(self.select_lane[junction])  # Number of lane actions
            n_actions_2 = 3  # Number of duration actions
            memory = ReplayMemory(max_size=10000, input_dims=input_dims, window_size=self.window_size)
            agent = BaseAgent(
                gamma=0.99,
                epsilon=0.3,
                lr=0.0001,
                input_dims=input_dims,
                n_actions_1=n_actions_1,
                n_actions_2=n_actions_2,
                batch_size=32,
                memory=memory,
                window_size=self.window_size,
                target_update_interval=10
            )
            agent.epsilon_dec = epsilon_decay
            self.agents[junction] = agent

    def get_incoming_edges(self, tl_id):
        edges_ids = list(set(lane_id[:-2] for lane_id in traci.trafficlight.getControlledLanes(tl_id)))
        return edges_ids

    def get_local_state(self, tl_id):
        incoming_edges = self.get_incoming_edges(tl_id)
        edge_features = []
        action_space = {0: 0, 3: 1, 6: 2, 9: 3}

        # Retrieve maximum vehicle counts for the current junction
        max_vehicle_counts = self.maximums.get(tl_id, None)

        # If maximums are not provided for this junction, set default maximums
        if max_vehicle_counts is None:
            max_vehicle_counts = np.array([50, 50, 50, 50])  # Default values
            logging.warning(f"No maximum vehicle counts provided for junction {tl_id}. Using default values.")

        # Ensure that we have maximum counts for each edge
        if len(max_vehicle_counts) < len(incoming_edges):
            # If fewer maximums are provided, pad with default values
            max_vehicle_counts = np.concatenate([
                max_vehicle_counts,
                np.full(len(incoming_edges) - len(max_vehicle_counts), 50)
            ])


        for idx, edge_id in enumerate(incoming_edges):
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            max_vehicle_count = max_vehicle_counts[idx]
            vehicle_count_norm = vehicle_count / max_vehicle_count  # Normalize using specific maximum

            if vehicle_count > 0:
                vehicle_watingtime = traci.edge.getWaitingTime(edge_id)
                mean_waitingtime = np.mean(vehicle_watingtime)
            else:
                mean_waitingtime = 0.0
            mean_waitingtime_norm = mean_waitingtime / 100  # Normalize

            edge_features.extend([vehicle_count_norm, mean_waitingtime_norm])

        # Ensure edge_features has a fixed size
        max_edges = 4  # Assuming a maximum of 4 incoming edges
        while len(edge_features) < 2 * max_edges:
            edge_features.extend([0.0, 0.0])  # Padding with zeros if necessary

        edge_features = np.array(edge_features[:2 * max_edges])

        # Create a one-hot encoded vector for the phase
        phase_vector = np.zeros(4)  # Assuming 4 possible phases
        current_phase = traci.trafficlight.getPhase(tl_id)
        phase_vector[action_space.get(current_phase, 0)] = 1

        # Concatenate the edge features and phase vector
        local_state = np.concatenate([edge_features, phase_vector])
        return local_state

    def get_junction_waiting_time(self, junction_id):
        edges_ids = self.get_incoming_edges(junction_id)
        total_waiting_time = 0
        vehicle_count = 0
        for edge_id in edges_ids:
            waiting_time = traci.edge.getWaitingTime(edge_id)
            num_vehicles = traci.edge.getLastStepVehicleNumber(edge_id)
            total_waiting_time += waiting_time
            vehicle_count += num_vehicles
        if vehicle_count == 0:
            return 0
        return total_waiting_time / vehicle_count

    def set_phase_duration(self, junction, duration, phase_code):
        traci.trafficlight.setRedYellowGreenState(junction, phase_code)
        traci.trafficlight.setPhaseDuration(junction, duration)

    def load_model(self, path):
            model_folder = os.path.join(self.base_path, "models")
            for junction in self.agents:
                model_path = os.path.join(model_folder, f"{junction}_final.weights.h5")
                if os.path.exists(model_path):
                    self.agents[junction].q_eval.model.load_weights(model_path)
                    print(f"Loaded weights for {junction} from {model_path}")
                else:
                    print(f"Model weights not found for {junction} at {model_path}")

    def run_simulation(self):
        avg_waiting_time_per_epoch = []
        total_rewards_per_epoch = []
        traci.start([checkBinary("sumo"), "-c", f"{self.base_path}/{self.config_path}"])
        all_junctions = traci.trafficlight.getIDList()
        self.initialize_agents(all_junctions)
        if not self.train:
            self.load_model(os.path.join(self.base_path, "models"))
        traci.close()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1} started.")
            total_reward = 0
            try:
                traci.start([checkBinary("sumo"), "-c", f"{self.base_path}/{self.config_path}", "--start", "--quit-on-end","--scale","0.5"])
                
                total_waiting_time = 0
                total_actions = 0
                exploration_count = 0
                exploitation_count = 0
                step = 0
                action_timers = {junction: 0 for junction in all_junctions}
                yellow_timers = {junction: 0 for junction in all_junctions}
                previous_states = {junction: [] for junction in all_junctions}
                previous_lane_actions = {junction: None for junction in all_junctions}
                previous_duration_actions = {junction: None for junction in all_junctions}
                previous_combined_metric = {junction: None for junction in all_junctions}
                previous_state_sequence = {junction: None for junction in all_junctions}

                while step < self.steps and traci.simulation.getMinExpectedNumber() > 0:
                    traci.simulationStep()
                    step += 1

                    for junction in all_junctions:
                        yellow_timers[junction] = max(0, yellow_timers[junction] - 1)
                        action_timers[junction] = max(0, action_timers[junction] - 1)

                        current_state = self.get_local_state(junction)

                        # Update the state history
                        if len(previous_states[junction]) == 0:
                            previous_states[junction] = [current_state] * self.window_size
                        else:
                            previous_states[junction].pop(0)
                            previous_states[junction].append(current_state)

                        if yellow_timers[junction] > 0:
                            continue

                        if yellow_timers[junction] == 0 and action_timers[junction] == 0:
                            if previous_lane_actions[junction] is not None:
                                new_state_sequence = np.array(previous_states[junction])
                                current_waiting_time = self.get_junction_waiting_time(junction)
                                current_combined_metric = current_waiting_time

                                if previous_combined_metric[junction] is not None:
                                    reward = previous_combined_metric[junction] - current_combined_metric
                                else:
                                    reward = 0  # No reward for the first action
                                total_reward += reward
                                done = step >= self.steps or traci.simulation.getMinExpectedNumber() == 0
                                self.agents[junction].memory.store_transition(
                                    previous_state_sequence[junction],
                                    previous_lane_actions[junction],
                                    previous_duration_actions[junction],
                                    reward,
                                    new_state_sequence,
                                    done
                                )
                                if self.train:
                                    self.agents[junction].learn()
                                previous_combined_metric[junction] = current_combined_metric

                            else:
                                current_waiting_time = self.get_junction_waiting_time(junction)
                                previous_combined_metric[junction] = current_waiting_time

                            current_state_sequence = np.array(previous_states[junction])
                            lane_action, duration_action, exploration = self.agents[junction].choose_action(current_state_sequence, self.train)

                            # Validate actions
                            lane_action = lane_action % len(self.select_lane[junction])
                            duration_action = duration_action % self.agents[junction].n_actions_2

                            exploration_count += int(exploration)
                            exploitation_count += int(not exploration)

                            yellow_phase_code = self.select_lane[junction][lane_action][1]
                            self.set_phase_duration(junction, 5, yellow_phase_code)
                            yellow_timers[junction] = 5

                            action_durations = [20, 35, 50]
                            action_duration = action_durations[duration_action]
                            action_timers[junction] = action_duration
                            previous_state_sequence[junction] = current_state_sequence.copy()
                            previous_lane_actions[junction] = lane_action
                            previous_duration_actions[junction] = duration_action

                        elif yellow_timers[junction] == 0 and action_timers[junction] > 0:
                            phase_code = self.select_lane[junction][previous_lane_actions[junction]][0]
                            self.set_phase_duration(junction, action_timers[junction], phase_code)
                            total_actions += 1

                        total_waiting_time += self.get_junction_waiting_time(junction)

                exploration_rate = exploration_count / (exploration_count + exploitation_count + 1e-5)
                print(f"Epoch {epoch + 1} ended. Exploration Rate: {exploration_rate:.2f}")

                avg_waiting_time = total_waiting_time / total_actions if total_actions > 0 else 0
                avg_waiting_time_per_epoch.append(avg_waiting_time)
                total_rewards_per_epoch.append(total_reward)
                print(f"Average Waiting Time in Epoch {epoch + 1}: {avg_waiting_time:.2f} seconds")
                traci.close()

                # Save checkpoints every 5 epochs
                if (epoch + 1) % 25 == 0:
                    checkpoint_folder = os.path.join(self.base_path, "models", "checkpoints")
                    if not os.path.exists(checkpoint_folder):
                        os.makedirs(checkpoint_folder)
                    for junction in all_junctions:
                        model_path = os.path.join(checkpoint_folder, f"{junction}_epoch_{epoch + 1}.weights.h5")
                        self.agents[junction].q_eval.model.save_weights(model_path)
                        print(f"Checkpoint for {junction} at epoch {epoch + 1} saved at {model_path}")

            except Exception as e:
                print(f"An error occurred during epoch {epoch + 1}: {e}")
                traci.close()
                raise e

        # Save final models
        model_folder = os.path.join(self.base_path, "models")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        for junction in all_junctions:
            model_path = os.path.join(model_folder, f"{junction}_final.weights.h5")
            self.agents[junction].q_eval.model.save_weights(model_path)
            print(f"Final model for {junction} saved at {model_path}")

        # Plotting results
        plot_folder = os.path.join(self.base_path, "plots")
        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        # Plot average waiting time
        plt.figure()
        plt.plot(range(1, self.epochs + 1), avg_waiting_time_per_epoch, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Average Waiting Time (seconds)')
        plt.title('Average Waiting Time Across Epochs')
        plt.grid(True)
        plot_path = os.path.join(plot_folder, 'average_waiting_time.png')
        plt.savefig(plot_path)
        print(f"Average waiting time plot saved at {plot_path}")
        plt.close()

        # Plot cumulative rewards
        plt.figure()
        plt.plot(range(1, self.epochs + 1), total_rewards_per_epoch, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Reward Across Epochs')
        plt.grid(True)
        reward_plot_path = os.path.join(plot_folder, 'cumulative_reward.png')
        plt.savefig(reward_plot_path)
        print(f"Cumulative reward plot saved at {reward_plot_path}")
        plt.close()

if __name__ == "__main__":
    config_path = r"conf/configuration.sumocfg"
    base_path = r"MARL"
    simulation = TrafficSimulation(config_path, base_path, epochs=50, steps=800, window_size=32,train=False)
    simulation.run_simulation()
