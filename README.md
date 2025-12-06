# CS3346-AI — Grid World Path Planning Using Markov Decision Processes

## Project Overview
This project implements a grid-world navigation system in which an autonomous agent (a vehicle) must reach a destination while avoiding obstacles and high-traffic roads. We model the decision-making process as a Markov Decision Process (MDP), where states correspond to road intersections, rewards represent traffic penalties, and terminal states represent successful arrival at the goal.

Value iteration is applied to compute an optimal policy, and the resulting route is visualized through a Python-based graphical interface.

This work was completed for the final project in **CS3346: Introduction to Artificial Intelligence** at Western University.

---

## Key Features
- CSV-based grid world input (roads, buildings, traffic values)
- MDP formulation with state, action, and reward modeling
- Value iteration algorithm to compute optimal policies
- Multiple environment configurations (normal layout, misleading shortcut, large grid)
- Visual simulation of the agent’s traversal

---

## Project Structure
/code
├── MDP.py # Core implementation of value iteration
└── visualizer.py # Rendering and simulation logic

/grids
├── grid.csv
├── grid1.csv
├── big_grid.csv
└── grid_bad_shortcut.csv

/resources
├── car.png
├── building.png
├── goal.png
└── road.png


---

## Running the Project

### Requirements
- Python 3.8+
- `pygame` or other required libraries (if applicable)

### Execution
1. Clone the repository  
2. Open /code directory  
3. Run
4. A window will open displaying the grid world.  
5. Select a map configuration from the menu or prompt.  
6. Observe the agent compute and follow the optimal path based on traffic penalties.


5. Select a map and observe the policy behavior.

---

## Team Members
- Evan Probst 
- Alan Su  
- Ethan Cawse 
- Aditya Vats  

---

## Status and Future Work
Current implementation uses value iteration. Planned extensions include:

- Reinforcement learning agent for policy learning
- Randomized transition probabilities
- Larger, more realistic urban maps

---

## Academic Context
Submitted for academic credit under CS3346 — Introduction to Artificial Intelligence (Fall 2025).


