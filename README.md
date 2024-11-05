# Circular Trajectory Generation with a Robot Manipulator - RX160

## Project Objective

This project aims to develop a motion primitive for the RX160 robot manipulator, allowing its end effector to follow a circular trajectory at a defined speed. The main goal is to link the operational and generalized spaces and simulate the trajectory in terms of position, velocity, and acceleration.

## Prerequisites

- Basic knowledge of robot modeling (direct and inverse kinematics)
- Python with the following libraries installed:
  - `numpy`
  - `matplotlib`
    
## Project Structure

### 1. Geometric Modeling 

The robot used in this project is a simplified version of the Staubli RX160 (RX160R), with 3 degrees of freedom (3R). You need to:

- Define the reference frames and DH parameters.
- Calculate the Direct Geometric Model (DGM).
- Implement a Python function to compute the DGM.

### 2. Differential Modeling

- Calculate the geometric and analytical Jacobian matrices.
- Implement a Python function for differential modeling calculations.

### 3. Motion Generation

The project includes generating circular trajectories with a speed profile defined by the user. The trajectory is determined by two points (A and B), which represent the diameter of the circle.

#### Steps:
- Calculate the operational trajectories: coordinates, velocities, and accelerations based on the curvilinear abscissa `s`.
- Compute time evolution laws to match the imposed speed profile.
- Generate motion in the joint space using inverse models.

### 4. Simulation

Trajectory simulation in the generalized space is carried out at each sampling instant with a time step `Te` (around 1 to 5 ms). This includes displaying trajectory curves, velocities, and accelerations in both operational and joint spaces.

## Key Features

- **Geometric modeling calculations**: Implement DGM and inverse geometric models.
- **Differential calculations**: Use Jacobian matrices to link the joint and operational spaces.
- **Circular trajectory generation**: Simulate imposed trajectories with customizable speed profiles.
- **Full simulation**: Visualize trajectories and validate the program with tests on different scenarios.

## Usage

1. Clone this project:
   ```bash
   git clone https://github.com/OlivierCrt/Engineering_Consultancy_Robotic_Modeling/tree/main

