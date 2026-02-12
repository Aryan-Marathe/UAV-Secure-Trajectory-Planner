import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# Explicit import for 3D plotting support
from mpl_toolkits.mplot3d import Axes3D 
from typing import Tuple, Dict, List, Optional
import pickle
import json
from datetime import datetime
import os


# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class UAVConfig:
    """Configuration parameters for UAV system"""

    def __init__(self):
        # Environment parameters
        self.area_size = 1000.0  # meters (square area)
        self.uav_height = 100.0  # meters
        self.max_velocity = 50.0  # m/s
        self.time_slots = 20  # Number of time slots
        self.slot_duration = 1.0  # seconds

        # Communication parameters
        self.carrier_freq = 2.4e9  # Hz
        self.bandwidth = 1e6  # Hz
        self.noise_power = 1e-10  # Watts
        self.path_loss_exponent = 2.0
        self.reference_distance = 1.0  # meters
        self.reference_loss = 30.0  # dB

        # Power parameters
        self.max_transmit_power = 1.0  # Watts
        self.min_transmit_power = 0.01  # Watts

        # Security parameters
        self.min_secrecy_rate = 1.0  # bits/s/Hz

        # Computation parameters
        self.cpu_cycles_per_bit = 1000  # cycles/bit
        self.uav_computation_capacity = 1e9  # cycles/second

        # ML Training parameters
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.train_test_split = 0.8

        # Constraint penalty weights
        self.penalty_mobility = 100.0
        self.penalty_power = 50.0
        self.penalty_secrecy = 200.0
        self.penalty_collision = 500.0
        self.penalty_boundary = 100.0


# ============================================================================
# CHANNEL MODEL AND PHYSICS
# ============================================================================

class ChannelModel:
    """Physical layer channel model for UAV communications"""

    def __init__(self, config: UAVConfig):
        self.config = config
        self.c = 3e8  # Speed of light
        self.wavelength = self.c / config.carrier_freq

    def compute_distance(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """Compute 3D Euclidean distance"""
        return np.sqrt(np.sum((uav_pos - ground_pos)**2))

    def compute_path_loss(self, distance: float) -> float:
        """Compute path loss in linear scale"""
        if distance < self.config.reference_distance:
            distance = self.config.reference_distance

        # Free space path loss model
        path_loss_db = self.config.reference_loss + \
                       10 * self.config.path_loss_exponent * np.log10(distance / self.config.reference_distance)

        return 10 ** (-path_loss_db / 10)

    def compute_channel_gain(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """Compute channel power gain"""
        distance = self.compute_distance(uav_pos, ground_pos)
        return self.compute_path_loss(distance)

    def compute_rate(self, channel_gain: float, transmit_power: float) -> float:
        """Compute achievable data rate (Shannon capacity)"""
        snr = (transmit_power * channel_gain) / self.config.noise_power
        return self.config.bandwidth * np.log2(1 + snr)

    def compute_secrecy_rate(self,
                            uav_pos: np.ndarray,
                            user_pos: np.ndarray,
                            eve_pos: np.ndarray,
                            transmit_power: float) -> float:
        """Compute secrecy rate (legitimate rate - eavesdropper rate)"""
        # Legitimate channel
        gain_user = self.compute_channel_gain(uav_pos, user_pos)
        rate_user = self.compute_rate(gain_user, transmit_power)

        # Eavesdropper channel
        gain_eve = self.compute_channel_gain(uav_pos, eve_pos)
        rate_eve = self.compute_rate(gain_eve, transmit_power)

        # Secrecy rate (positive part)
        return max(0, rate_user - rate_eve)


# ============================================================================
# DATASET GENERATION
# ============================================================================

class TrajectoryDataset(Dataset):
    """Dataset for training ML-based trajectory planner"""

    def __init__(self,
                 scenarios: List[Dict],
                 trajectories: List[np.ndarray],
                 powers: List[np.ndarray],
                 objectives: List[float]):
        """
        Args:
            scenarios: List of scenario dictionaries with user/eavesdropper positions
            trajectories: List of optimal trajectories from optimization
            powers: List of optimal power allocations
            objectives: List of objective values achieved
        """
        self.scenarios = scenarios
        self.trajectories = trajectories
        self.powers = powers
        self.objectives = objectives

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]

        # Input features: flatten all positions
        user_pos = scenario['user_position']
        eve_pos = scenario['eavesdropper_position']
        initial_pos = scenario['initial_uav_position']

        # Concatenate all input features
        features = np.concatenate([
            user_pos.flatten(),
            eve_pos.flatten(),
            initial_pos.flatten()
        ])

        # Output: trajectory and power allocation
        trajectory = self.trajectories[idx]
        power = self.powers[idx]

        return (torch.FloatTensor(features),
                torch.FloatTensor(trajectory.flatten()),
                torch.FloatTensor(power),
                torch.FloatTensor([self.objectives[idx]]))


def generate_random_scenario(config: UAVConfig) -> Dict:
    """Generate a random scenario with user, eavesdropper, and initial UAV position"""

    # Random user position (2D ground)
    user_pos = np.array([
        np.random.uniform(0, config.area_size),
        np.random.uniform(0, config.area_size),
        0.0  # ground level
    ])

    # Random eavesdropper position (ensure some distance from user)
    min_distance = 50.0
    while True:
        eve_pos = np.array([
            np.random.uniform(0, config.area_size),
            np.random.uniform(0, config.area_size),
            0.0
        ])
        if np.linalg.norm(user_pos[:2] - eve_pos[:2]) > min_distance:
            break

    # Random initial UAV position
    initial_uav = np.array([
        np.random.uniform(0, config.area_size),
        np.random.uniform(0, config.area_size),
        config.uav_height
    ])

    return {
        'user_position': user_pos,
        'eavesdropper_position': eve_pos,
        'initial_uav_position': initial_uav
    }


def generate_optimization_solution(scenario: Dict, config: UAVConfig) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generate approximate optimization solution using heuristic approach.
    In practice, this would call the actual SCA optimization solver.
    """
    user_pos = scenario['user_position']
    eve_pos = scenario['eavesdropper_position']
    initial_pos = scenario['initial_uav_position']

    trajectory = np.zeros((config.time_slots, 3))
    powers = np.zeros(config.time_slots)

    # Heuristic: Move towards a position that maximizes distance from eavesdropper
    # while staying reasonably close to user

    # Target position: weighted center closer to user, away from eavesdropper
    target_2d = 0.7 * user_pos[:2] + 0.3 * (2 * user_pos[:2] - eve_pos[:2])
    target_2d = np.clip(target_2d, 0, config.area_size)
    target_pos = np.array([target_2d[0], target_2d[1], config.uav_height])

    # Generate smooth trajectory
    for t in range(config.time_slots):
        alpha = t / (config.time_slots - 1)
        trajectory[t] = (1 - alpha) * initial_pos + alpha * target_pos

        # Ensure height constraint
        trajectory[t, 2] = config.uav_height

        # Compute power allocation based on distance to user
        channel = ChannelModel(config)
        dist_to_user = channel.compute_distance(trajectory[t], user_pos)

        # Power inversely proportional to channel quality (water-filling approximation)
        powers[t] = np.clip(
            config.max_transmit_power * (dist_to_user / 200.0),
            config.min_transmit_power,
            config.max_transmit_power
        )

    # Compute objective (sum secrecy rate)
    channel = ChannelModel(config)
    total_secrecy_rate = 0.0
    for t in range(config.time_slots):
        secrecy_rate = channel.compute_secrecy_rate(
            trajectory[t], user_pos, eve_pos, powers[t]
        )
        total_secrecy_rate += secrecy_rate

    return trajectory, powers, total_secrecy_rate


def generate_training_data(config: UAVConfig, num_samples: int = 1000) -> TrajectoryDataset:
    """Generate training dataset by solving optimization problems"""

    print(f"Generating {num_samples} training samples...")

    scenarios = []
    trajectories = []
    powers = []
    objectives = []

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

        # Generate random scenario
        scenario = generate_random_scenario(config)

        # Solve optimization (or use heuristic)
        trajectory, power, objective = generate_optimization_solution(scenario, config)

        scenarios.append(scenario)
        trajectories.append(trajectory)
        powers.append(power)
        objectives.append(objective)

    print("Training data generation complete!")

    return TrajectoryDataset(scenarios, trajectories, powers, objectives)


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class TrajectoryPlannerNetwork(nn.Module):
    """
    Deep neural network for end-to-end trajectory and power planning.

    Architecture:
    - Input: Scenario features (user, eavesdropper, initial UAV positions)
    - Output: UAV trajectory (T x 3) and power allocation (T x 1)
    - Uses separate heads for trajectory and power prediction
    - Includes residual connections for better gradient flow
    """

    def __init__(self, config: UAVConfig):
        super(TrajectoryPlannerNetwork, self).__init__()

        self.config = config
        self.time_slots = config.time_slots

        # Input dimension: 3 positions × 3 coordinates = 9
        input_dim = 9

        # Output dimensions
        trajectory_dim = config.time_slots * 3  # T × (x, y, z)
        power_dim = config.time_slots  # T × power

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Trajectory prediction head
        self.trajectory_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, trajectory_dim)
        )

        # Power allocation head
        self.power_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, power_dim),
            nn.Sigmoid()  # Output in [0, 1], will be scaled to power range
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input features [batch_size, 9]

        Returns:
            trajectory: [batch_size, time_slots, 3]
            power: [batch_size, time_slots]
        """
        # Extract shared features
        features = self.feature_extractor(x)

        # Predict trajectory
        trajectory_flat = self.trajectory_head(features)
        trajectory = trajectory_flat.view(-1, self.time_slots, 3)

        # Predict power allocation
        power_normalized = self.power_head(features)
        power = (
            self.config.min_transmit_power +
            power_normalized * (self.config.max_transmit_power - self.config.min_transmit_power))

        return trajectory, power


# ============================================================================
# CONSTRAINT ENFORCEMENT AND LOSS FUNCTIONS
# ============================================================================

class ConstraintPenalties:
    """Compute penalty terms for constraint violations"""

    def __init__(self, config: UAVConfig):
        self.config = config
        self.channel = ChannelModel(config)

    def mobility_constraint_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Penalize violations of maximum velocity constraint.

        Args:
            trajectory: [batch_size, time_slots, 3]
        """
        # Compute velocities between consecutive time slots
        velocities = trajectory[:, 1:, :] - trajectory[:, :-1, :]
        speeds = torch.norm(velocities, dim=2) / self.config.slot_duration

        # Penalty for exceeding max velocity
        violations = torch.relu(speeds - self.config.max_velocity)
        return torch.mean(violations ** 2)

    def power_constraint_penalty(self, power: torch.Tensor) -> torch.Tensor:
        """
        Penalize violations of power limits.

        Args:
            power: [batch_size, time_slots]
        """
        # Power should already be in valid range due to sigmoid + scaling
        # But add penalty for safety
        lower_violations = torch.relu(self.config.min_transmit_power - power)
        upper_violations = torch.relu(power - self.config.max_transmit_power)

        return torch.mean(lower_violations ** 2 + upper_violations ** 2)

    def boundary_constraint_penalty(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Penalize UAV going outside operational area.

        Args:
            trajectory: [batch_size, time_slots, 3]
        """
        # X and Y coordinates should be in [0, area_size]
        x_lower = torch.relu(-trajectory[:, :, 0])
        x_upper = torch.relu(trajectory[:, :, 0] - self.config.area_size)
        y_lower = torch.relu(-trajectory[:, :, 1])
        y_upper = torch.relu(trajectory[:, :, 1] - self.config.area_size)

        # Height should be constant
        height_violation = torch.abs(trajectory[:, :, 2] - self.config.uav_height)

        boundary_violations = x_lower + x_upper + y_lower + y_upper + height_violation
        return torch.mean(boundary_violations ** 2)

    def secrecy_rate_penalty(self,
                            trajectory: torch.Tensor,
                            power: torch.Tensor,
                            user_pos: np.ndarray,
                            eve_pos: np.ndarray) -> torch.Tensor:
        """
        Penalize insufficient secrecy rate (computed in numpy for simplicity).

        Args:
            trajectory: [batch_size, time_slots, 3]
            power: [batch_size, time_slots]
            user_pos: [batch_size, 3]
            eve_pos: [batch_size, 3]
        """
        batch_size = trajectory.shape[0]
        penalties = []

        # Convert to numpy for channel calculations
        traj_np = trajectory.detach().cpu().numpy()
        power_np = power.detach().cpu().numpy()

        for b in range(batch_size):
            total_penalty = 0.0
            for t in range(self.config.time_slots):
                secrecy_rate = self.channel.compute_secrecy_rate(
                    traj_np[b, t, :],
                    user_pos[b],
                    eve_pos[b],
                    power_np[b, t]
                )

                # Penalty if secrecy rate below minimum
                if secrecy_rate < self.config.min_secrecy_rate:
                    total_penalty += (self.config.min_secrecy_rate - secrecy_rate) ** 2

            penalties.append(total_penalty)

        return torch.tensor(penalties, device=trajectory.device).mean()

    def collision_avoidance_penalty(self,
                                   trajectory: torch.Tensor,
                                   user_pos: np.ndarray,
                                   eve_pos: np.ndarray,
                                   min_distance: float = 20.0) -> torch.Tensor:
        """
        Penalize UAV getting too close to ground positions.

        Args:
            trajectory: [batch_size, time_slots, 3]
            user_pos: [batch_size, 3]
            eve_pos: [batch_size, 3]
            min_distance: Minimum safe distance
        """
        batch_size = trajectory.shape[0]
        penalties = []

        traj_np = trajectory.detach().cpu().numpy()

        for b in range(batch_size):
            total_penalty = 0.0
            for t in range(self.config.time_slots):
                # Distance to user
                dist_user = np.linalg.norm(traj_np[b, t, :] - user_pos[b])
                if dist_user < min_distance:
                    total_penalty += (min_distance - dist_user) ** 2

                # Distance to eavesdropper
                dist_eve = np.linalg.norm(traj_np[b, t, :] - eve_pos[b])
                if dist_eve < min_distance:
                    total_penalty += (min_distance - dist_eve) ** 2

            penalties.append(total_penalty)

        return torch.tensor(penalties, device=trajectory.device).mean()


class UAVLoss(nn.Module):
    """
    Combined loss function for UAV trajectory planning.

    Components:
    1. Prediction loss (MSE with optimal solutions)
    2. Physical constraint penalties
    3. Security constraint penalties
    """

    def __init__(self, config: UAVConfig):
        super(UAVLoss, self).__init__()
        self.config = config
        self.constraints = ConstraintPenalties(config)
        self.mse = nn.MSELoss()

    def forward(self,
                pred_trajectory: torch.Tensor,
                pred_power: torch.Tensor,
                target_trajectory: torch.Tensor,
                target_power: torch.Tensor,
                user_positions: np.ndarray,
                eve_positions: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss with all components.

        Returns:
            total_loss: Scalar loss value
            loss_dict: Dictionary with individual loss components
        """
        # Prediction losses
        trajectory_loss = self.mse(pred_trajectory, target_trajectory)
        power_loss = self.mse(pred_power, target_power)

        # Constraint penalties
        mobility_penalty = self.constraints.mobility_constraint_penalty(pred_trajectory)
        power_penalty = self.constraints.power_constraint_penalty(pred_power)
        boundary_penalty = self.constraints.boundary_constraint_penalty(pred_trajectory)
        secrecy_penalty = self.constraints.secrecy_rate_penalty(
            pred_trajectory, pred_power, user_positions, eve_positions
        )
        collision_penalty = self.constraints.collision_avoidance_penalty(
            pred_trajectory, user_positions, eve_positions
        )

        # Weighted total loss
        total_loss = (
            trajectory_loss +
            power_loss +
            self.config.penalty_mobility * mobility_penalty +
            self.config.penalty_power * power_penalty +
            self.config.penalty_boundary * boundary_penalty +
            self.config.penalty_secrecy * secrecy_penalty +
            self.config.penalty_collision * collision_penalty
        )

        # Return loss components for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'trajectory': trajectory_loss.item(),
            'power': power_loss.item(),
            'mobility': mobility_penalty.item(),
            'power_constraint': power_penalty.item(),
            'boundary': boundary_penalty.item(),
            'secrecy': secrecy_penalty.item(),
            'collision': collision_penalty.item()
        }

        return total_loss, loss_dict


# ============================================================================
# TRAINING LOOP
# ============================================================================

class UAVTrainer:
    """Trainer for UAV trajectory planning network"""

    def __init__(self, config: UAVConfig, model: TrajectoryPlannerNetwork):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.criterion = UAVLoss(config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        for batch_idx, (features, target_traj, target_power, _) in enumerate(dataloader):
            # Move to device
            features = features.to(self.device)
            target_traj = target_traj.view(-1, self.config.time_slots, 3).to(self.device)
            target_power = target_power.to(self.device)

            # Extract user and eve positions from features
            batch_size = features.shape[0]
            user_positions = features[:, :3].cpu().numpy()
            eve_positions = features[:, 3:6].cpu().numpy()

            # Forward pass
            pred_traj, pred_power = self.model(features)

            # Compute loss
            loss, loss_dict = self.criterion(
                pred_traj, pred_power,
                target_traj, target_power,
                user_positions, eve_positions
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict)

        # Average losses
        avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_losses

    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate on validation set"""
        self.model.eval()
        epoch_losses = []

        with torch.no_grad():
            for features, target_traj, target_power, _ in dataloader:
                features = features.to(self.device)
                target_traj = target_traj.view(-1, self.config.time_slots, 3).to(self.device)
                target_power = target_power.to(self.device)

                batch_size = features.shape[0]
                user_positions = features[:, :3].cpu().numpy()
                eve_positions = features[:, 3:6].cpu().numpy()

                pred_traj, pred_power = self.model(features)

                loss, loss_dict = self.criterion(
                    pred_traj, pred_power,
                    target_traj, target_power,
                    user_positions, eve_positions
                )

                epoch_losses.append(loss_dict)

        avg_losses = {k: np.mean([d[k] for d in epoch_losses]) for k in epoch_losses[0].keys()}
        return avg_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Full training loop"""
        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*60}")

            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_losses.append(train_losses)

            # Validate
            val_losses = self.validate(val_loader)
            self.val_losses.append(val_losses)

            # Update learning rate
            self.scheduler.step(val_losses['total'])

            # Print losses
            print(f"\nTraining Losses:")
            for k, v in train_losses.items():
                print(f"  {k:20s}: {v:.6f}")

            print(f"\nValidation Losses:")
            for k, v in val_losses.items():
                print(f"  {k:20s}: {v:.6f}")

            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                self.save_checkpoint('/home/sandbox/best_uav_model.pth')
                print(f"\n✓ Saved new best model (val_loss: {best_val_loss:.6f})")

        print(f"\n{'='*60}")
        print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
        print(f"{'='*60}\n")

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']


# ============================================================================
# INFERENCE AND DEPLOYMENT
# ============================================================================

class UAVDeployment:
    """Real-time deployment interface for trained model"""

    def __init__(self, model_path: str, config: UAVConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load trained model
        self.model = TrajectoryPlannerNetwork(config)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.channel = ChannelModel(config)

        print(f"Model loaded successfully on {self.device}")

    def predict(self,
                user_position: np.ndarray,
                eavesdropper_position: np.ndarray,
                initial_uav_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Real-time trajectory and power prediction.

        Args:
            user_position: [3] array
            eavesdropper_position: [3] array
            initial_uav_position: [3] array

        Returns:
            trajectory: [time_slots, 3] array
            power: [time_slots] array
            metrics: Dictionary with performance metrics
        """
        # Prepare input
        features = np.concatenate([
            user_position.flatten(),
            eavesdropper_position.flatten(),
            initial_uav_position.flatten()
        ])

        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            pred_traj, pred_power = self.model(features_tensor)

        # Convert to numpy
        trajectory = pred_traj.cpu().numpy()[0]
        power = pred_power.cpu().numpy()[0]

        # Compute metrics
        metrics = self.compute_metrics(trajectory, power, user_position, eavesdropper_position)

        return trajectory, power, metrics

    def compute_metrics(self,
                       trajectory: np.ndarray,
                       power: np.ndarray,
                       user_pos: np.ndarray,
                       eve_pos: np.ndarray) -> Dict:
        """Compute performance metrics for predicted solution"""

        metrics = {
            'total_secrecy_rate': 0.0,
            'avg_secrecy_rate': 0.0,
            'min_secrecy_rate': float('inf'),
            'max_velocity': 0.0,
            'avg_power': 0.0,
            'constraint_violations': 0
        }

        secrecy_rates = []

        for t in range(self.config.time_slots):
            # Secrecy rate
            secrecy_rate = self.channel.compute_secrecy_rate(
                trajectory[t], user_pos, eve_pos, power[t]
            )
            secrecy_rates.append(secrecy_rate)
            metrics['total_secrecy_rate'] += secrecy_rate
            metrics['min_secrecy_rate'] = min(metrics['min_secrecy_rate'], secrecy_rate)

            # Check velocity constraint
            if t > 0:
                velocity = np.linalg.norm(trajectory[t] - trajectory[t-1]) / self.config.slot_duration
                metrics['max_velocity'] = max(metrics['max_velocity'], velocity)
                if velocity > self.config.max_velocity:
                    metrics['constraint_violations'] += 1

        metrics['avg_secrecy_rate'] = metrics['total_secrecy_rate'] / self.config.time_slots
        metrics['avg_power'] = np.mean(power)
        metrics['secrecy_rates'] = secrecy_rates

        return metrics

    def visualize_solution(self,
                          trajectory: np.ndarray,
                          power: np.ndarray,
                          user_pos: np.ndarray,
                          eve_pos: np.ndarray,
                          metrics: Dict,
                          save_path: str = '/home/sandbox/trajectory_visualization.png'):
        """Visualize the predicted trajectory and metrics with enhanced 3D path"""

        fig = plt.figure(figsize=(16, 10))

        # --- MODIFIED 3D PLOT ---
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # 1. Plot the main trajectory line
        ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                'b-', linewidth=2, alpha=0.7, label='Flight Path')

        # 2. Add directional arrows (quivers) between time steps
        # Compute vectors between consecutive points
        x = trajectory[:-1, 0]
        y = trajectory[:-1, 1]
        z = trajectory[:-1, 2]
        u = trajectory[1:, 0] - x
        v = trajectory[1:, 1] - y
        w = trajectory[1:, 2] - z
        
        # Plot arrows
        ax1.quiver(x, y, z, u, v, w, length=1.0, normalize=False, 
                  color='deepskyblue', arrow_length_ratio=0.3, alpha=0.8)

        # 3. Add "Ground Shadow" (Projection on Z=0) for depth perception
        ax1.plot(trajectory[:, 0], trajectory[:, 1], np.zeros_like(trajectory[:, 2]),
                color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Ground Shadow')

        # 4. Markers for User, Eve, Start, and End
        ax1.scatter(*user_pos, c='green', s=200, marker='^', label='User', edgecolors='black', depthshade=False)
        ax1.scatter(*eve_pos, c='red', s=200, marker='v', label='Eavesdropper', edgecolors='black', depthshade=False)
        ax1.scatter(*trajectory[0], c='blue', s=100, marker='s', label='Start', edgecolors='white')
        ax1.scatter(*trajectory[-1], c='purple', s=150, marker='*', label='End', edgecolors='white')

        # Labels and Limits
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Height (m)')
        ax1.set_title('3D UAV Path with Direction')
        ax1.legend(loc='upper left', fontsize='small')
        ax1.grid(True)
        # ------------------------

        # 2D top view
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=6)
        
        # Add arrows to 2D view as well
        for i in range(len(trajectory)-1):
            ax2.arrow(trajectory[i,0], trajectory[i,1], 
                     trajectory[i+1,0]-trajectory[i,0], trajectory[i+1,1]-trajectory[i,1],
                     head_width=15, head_length=20, fc='cyan', ec='cyan', alpha=0.6)

        ax2.scatter(user_pos[0], user_pos[1], c='green', s=200, marker='^',
                   label='User', edgecolors='black', zorder=5)
        ax2.scatter(eve_pos[0], eve_pos[1], c='red', s=200, marker='v',
                   label='Eavesdropper', edgecolors='black', zorder=5)
        ax2.scatter(trajectory[0, 0], trajectory[0, 1], c='blue', s=150,
                   marker='s', label='Start', edgecolors='black', zorder=5)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('2D Top View')
        ax2.legend()
        ax2.grid(True)
        ax2.set_aspect('equal')

        # Power allocation over time
        ax3 = fig.add_subplot(2, 3, 3)
        time_slots = np.arange(len(power))
        ax3.plot(time_slots, power, 'g-o', linewidth=2, markersize=6)
        ax3.axhline(self.config.max_transmit_power, color='r', linestyle='--', 
                   label=f'Max Power ({self.config.max_transmit_power}W)')
        ax3.axhline(self.config.min_transmit_power, color='orange', linestyle='--', 
                   label=f'Min Power ({self.config.min_transmit_power}W)')
        ax3.set_xlabel('Time Slot')
        ax3.set_ylabel('Power (W)')
        ax3.set_title('Power Allocation')
        ax3.legend()
        ax3.grid(True)

        # Secrecy rate over time
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(time_slots, metrics['secrecy_rates'], 'purple', linewidth=2, marker='o')
        ax4.axhline(self.config.min_secrecy_rate, color='r', linestyle='--', 
                   label=f'Min Required ({self.config.min_secrecy_rate} bits/s/Hz)')
        ax4.set_xlabel('Time Slot')
        ax4.set_ylabel('Secrecy Rate (bits/s/Hz)')
        ax4.set_title('Secrecy Rate Over Time')
        ax4.legend()
        ax4.grid(True)

        # Velocity profile
        ax5 = fig.add_subplot(2, 3, 5)
        velocities = []
        for t in range(1, len(trajectory)):
            vel = np.linalg.norm(trajectory[t] - trajectory[t-1]) / self.config.slot_duration
            velocities.append(vel)
        ax5.plot(range(1, len(trajectory)), velocities, 'orange', linewidth=2, marker='o')
        ax5.axhline(self.config.max_velocity, color='r', linestyle='--', 
                   label=f'Max Velocity ({self.config.max_velocity} m/s)')
        ax5.set_xlabel('Time Slot')
        ax5.set_ylabel('Velocity (m/s)')
        ax5.set_title('UAV Velocity Profile')
        ax5.legend()
        ax5.grid(True)

        # Metrics summary
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        summary_text = f"""
        PERFORMANCE METRICS
        {'='*40}

        Total Secrecy Rate: {metrics['total_secrecy_rate']:.2f} bits/s/Hz
        Average Secrecy Rate: {metrics['avg_secrecy_rate']:.2f} bits/s/Hz
        Minimum Secrecy Rate: {metrics['min_secrecy_rate']:.2f} bits/s/Hz

        Maximum Velocity: {metrics['max_velocity']:.2f} m/s
        Average Power: {metrics['avg_power']:.4f} W

        Constraint Violations: {metrics['constraint_violations']}

        {'='*40}
        Status: {'✓ FEASIBLE' if metrics['constraint_violations'] == 0 else '✗ INFEASIBLE'}
        """
        ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

        return fig


# ============================================================================
# MAIN EXECUTION AND EXAMPLE USAGE
# ============================================================================

def main():
    """Main execution pipeline"""

    print("="*60)
    print("ML-Based UAV Trajectory Design for Secure Computation")
    print("="*60)

    # Initialize configuration
    config = UAVConfig()

    # Step 1: Generate training data
    print("\n[STEP 1] Generating Training Data...")
    dataset = generate_training_data(config, num_samples=1000)

    # Split into train and validation
    train_size = int(config.train_test_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")

    # Step 2: Initialize model
    print("\n[STEP 2] Initializing Neural Network...")
    model = TrajectoryPlannerNetwork(config)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 3: Train model
    print("\n[STEP 3] Training Model...")
    trainer = UAVTrainer(config, model)
    trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)

    # Step 4: Test deployment
    print("\n[STEP 4] Testing Real-Time Deployment...")
    deployment = UAVDeployment('/home/sandbox/best_uav_model.pth', config)

    # Generate test scenario
    test_scenario = generate_random_scenario(config)

    print("\nTest Scenario:")
    print(f"  User position: {test_scenario['user_position']}")
    print(f"  Eavesdropper position: {test_scenario['eavesdropper_position']}")
    print(f"  Initial UAV position: {test_scenario['initial_uav_position']}")

    # Predict trajectory
    trajectory, power, metrics = deployment.predict(
        test_scenario['user_position'],
        test_scenario['eavesdropper_position'],
        test_scenario['initial_uav_position']
    )

    print("\nPrediction Results:")
    print(f"  Total secrecy rate: {metrics['total_secrecy_rate']:.2f} bits/s/Hz")
    print(f"  Average secrecy rate: {metrics['avg_secrecy_rate']:.2f} bits/s/Hz")
    print(f"  Maximum velocity: {metrics['max_velocity']:.2f} m/s")
    print(f"  Constraint violations: {metrics['constraint_violations']}")

    # Visualize
    deployment.visualize_solution(
        trajectory, power,
        test_scenario['user_position'],
        test_scenario['eavesdropper_position'],
        metrics
    )

    # Step 5: Save final artifacts
    print("\n[STEP 5] Saving Artifacts...")

    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'config': config.__dict__
    }
    
    # Ensure the directory exists before saving
    output_dir = '/home/sandbox'
    os.makedirs(output_dir, exist_ok=True)

    with open('/home/sandbox/training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print("  ✓ Training history saved")
    print("  ✓ Best model saved")
    print("  ✓ Visualization saved")

    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)

    return deployment, metrics


if __name__ == "__main__":
    deployment, metrics = main()