# Agent-Based Sorting Simulation

An agent-based simulation of self-organized object clustering and sorting on a two-dimensional grid. The project investigates how local interactions between mobile agents and objects can lead to spatial patterns without centralized control.

## Overview

This project implements an agent-based model in Python. Agents move randomly across a periodic grid and pick up or drop objects according to local object density and probabilistic rules.

The simulation is designed to explore how different agent populations and interaction rules influence the formation of clusters, the separation of object types, and the spatial organization of the system.

The repository contains two simulation scripts and a module for quantitative cluster analysis.

## Features

- **Agent-based modeling:** Agents move independently and interact with objects in their local environment.
- **Probabilistic pickup and drop-off:** Decisions depend on local object density and configurable parameters.
- **Periodic boundary conditions:** The grid wraps around at its edges, forming a toroidal space.
- **Multiple object types:** The extended simulation distinguishes red and blue objects.
- **Heterogeneous agents:** Agents can interact with red objects, blue objects, or both.
- **Cluster analysis:** Measure cluster counts, color composition, spatial compactness, and spatial entropy.
- **Data export:** Simulation results can be saved as CSV files for further analysis.
- **Visualization:** Grid configurations can be exported as PNG images.

## Repository Structure

```text
.
├── Simulation.py
├── Simulation_MultiAgent.py
├── ClusteringMaße.py
├── README.md
└── requirements.txt
```

### 1. `Simulation.py`

A basic simulation according to the model of Deneubourg et al. (1999) with mobile agents and a single object type.

Main features:
- Random agent movement
- Probabilistic pickup and drop-off
- Local neighborhood density
- Cluster counting and spatial entropy
- Periodic visualization of the grid

### 2. `Simulation_MultiAgent.py`

An extended simulation with two object types and three agent types.

| Agent type | Behavior |
|---|---|
| Type A | Interacts with red objects |
| Type B | Interacts with blue objects |
| Type AB | Can interact with both object types |

The script evaluates different agent population configurations and repeats each configuration multiple times.

Results are exported to a CSV file, and grid snapshots can be saved as PNG images.

### 3. `ClusteringMaße.py`

A module containing functions for quantitative analysis of spatial patterns.

The module includes:

- **Cluster detection:** Identifies connected groups of occupied cells using a Manhattan neighborhood.
- **Cluster counting:** Counts all clusters exceeding a specified size threshold.
- **Color-specific cluster counting:** Counts clusters containing more than a specified number of red or blue objects.
- **Spatial entropy:** Measures the spatial distribution of occupied cells across grid patches.
- **Cluster compactness:** Calculates weighted mean distances of red and blue objects from a circular cluster center.
- **Color composition:** Identifies predominantly red, predominantly blue, and mixed clusters.
- **Weighted color ratios:** Calculates the weighted red-to-blue ratio of mixed clusters.

The grid uses the following object encoding:

| Value | Meaning |
|---|---|
| `0` | Empty cell |
| `1` | Red object |
| `2` | Blue object |

Cluster detection uses periodic boundary conditions and a Manhattan-distance neighborhood.

## Model Description

### Agent movement

At each simulation step, every agent randomly chooses one of four possible directions: up, down, left, or right.

An agent moves only if its destination cell is not occupied by another agent. Movement across a grid boundary wraps around to the opposite side.


### Local neighborhood density

The local density $f$ is calculated as the proportion of cells containing relevant objects within a Manhattan-distance neighborhood.

For radius $R$, the neighborhood consists of cells satisfying

$$
|dx| + |dy| \leq R.
$$

For heterogeneous agents, the relevant objects depend on the agent type.

### Pickup and drop-off

The probabilities are defined as

$$
P_{\mathrm{pickup}} =
\left(\frac{K_+}{K_+ + f}\right)^2
$$

and

$$
P_{\mathrm{drop}} =
\left(\frac{f}{K_- + f}\right)^2.
$$

Here, $K_+$ and $K_-$ are model parameters controlling pickup and drop-off behavior, and $f$ is the local density of relevant objects.

These rules make pickup more likely at low local density and drop-off more likely at high local density, supporting the emergence of spatial clustering.

## Installation

Python 3.10 or newer is recommended.

Clone the repository:

```bash
git clone https://github.com/MaAntBrum/ant_based_clustering_variation.git
cd ant_based_clustering_variation
```

Install the required dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

This command installs all required Python packages listed in `requirements.txt`.

The cluster analysis module also uses Python's standard library, including `functools` and `dataclasses`.



## Running the Simulation

Run the basic simulation:

```bash
python Simulation.py
```

Run the multi-agent simulation:

```bash
python Simulation_MultiAgent.py
```

Make sure `ClusteringMaße.py` is located in the same directory as the simulation scripts.

**Important:** The multi-agent script uses long simulation runs and multiple parameter configurations. The default settings can require substantial computation time and produce many output files. For initial testing, reduce the number of simulation steps and runs.

For example:

```python
RUNS_PER_PARAM = 1
```

and reduce the `STEPS` value in the parameter configurations.

## Configuration

The simulations can be adapted by changing parameters such as:

| Parameter | Description |
|---|---|
| `GRID_SIZE` | Width and height of the grid |
| `NUM_AGENTS` | Number of agents in the basic simulation |
| `NUM_AGENTS_A` | Number of agents interacting with red objects |
| `NUM_AGENTS_B` | Number of agents interacting with blue objects |
| `NUM_AGENTS_AB` | Number of agents interacting with both object types |
| `NUM_OBJECTS_A` | Number of red objects |
| `NUM_OBJECTS_B` | Number of blue objects |
| `STEPS` | Number of simulation steps |
| `RADIUS` | Radius of the local Manhattan neighborhood |
| `K_PLUS` | Pickup probability parameter |
| `K_MINUS` | Drop-off probability parameter |
| `RUNS_PER_PARAM` | Number of repetitions per parameter configuration |

The parameter values should be adjusted according to the research question and available computing resources.

## Output

Depending on the selected simulation, the program produces:

- Console output containing simulation progress and cluster statistics
- PNG images of grid configurations
- CSV files containing measurements for individual runs and averages across repeated runs

The CSV output can be analyzed with Python, R, Excel, or other data analysis tools.

## Limitations and Notes

- The simulations use stochastic movement and probabilistic interactions. Results may vary between runs.
- Random seeds are not currently fixed, so runs are not guaranteed to be reproducible.
- Cluster statistics depend on the neighborhood radius and minimum cluster-size threshold.
- Cluster detection connects occupied cells regardless of color. Color-specific cluster counts therefore measure color composition within connected components rather than independently connected components of each color.
- The spatial entropy implementation uses a particular patch-based definition. Its values should be interpreted consistently when comparing configurations.
- Large parameter sweeps can be computationally expensive.

## Author

**Marc-Antony Brumme**

GitHub: [MaAntBrum](https://github.com/YOUR-USERNAME)
