import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Vor dem Import von pyplot!
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
import ClusteringMaße as cm


# PARAMETER
#######################

GRID_SIZE = 100
RADIUS = 2
K_PLUS = 0.05
K_MINUS = 0.3
#VISUALIZE_EVERY = 5000000
RUNS_PER_PARAM = 10

parameter_sets = [
    {"NUM_AGENTS_A": 12, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 11, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 10, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 9, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 3, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 8, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 4, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},

    {"NUM_AGENTS_A": 11, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 10, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 9, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 8, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 3, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},

    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 4, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 5, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 6, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    
    {"NUM_AGENTS_A": 10, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 9, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 8, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 3, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},

    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 5, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 6, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 7, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    
    {"NUM_AGENTS_A": 9, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 8, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 3, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 4, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 4, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 5, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 3, "NUM_AGENTS_B": 3, "NUM_AGENTS_AB": 6, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 5, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 5, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 5, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},

    {"NUM_AGENTS_A": 8, "NUM_AGENTS_B": 4, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 7, "NUM_AGENTS_B": 4, "NUM_AGENTS_AB": 1, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 4, "NUM_AGENTS_AB": 2, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 4, "NUM_AGENTS_AB": 3, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 4, "NUM_AGENTS_B": 4, "NUM_AGENTS_AB": 4, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 6, "NUM_AGENTS_AB": 0, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},

    {"NUM_AGENTS_A": 4, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 7, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 3, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 8, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 2, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 9, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 1, "NUM_AGENTS_B": 1, "NUM_AGENTS_AB": 10, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 6, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 4, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 5, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 5, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 4, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 6, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 3, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 7, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 2, "NUM_AGENTS_B": 2, "NUM_AGENTS_AB": 8, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
   
    {"NUM_AGENTS_A": 4, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 8, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 3, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 9, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 2, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 10, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001},
    {"NUM_AGENTS_A": 1, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 11, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}, 
    {"NUM_AGENTS_A": 0, "NUM_AGENTS_B": 0, "NUM_AGENTS_AB": 12, "NUM_OBJECTS_A": 200, "NUM_OBJECTS_B": 200, "STEPS" : 15000001}

    # ... weitere Parametersätze
]


# Agentenklasse
###################

class Agent:
    def __init__(self, x, y, agent_type):
        self.x = x
        self.y = y
        self.carrying = 0
        self.agent_type = agent_type  # 1=rot, 2=blau, 3=beide


# Hilfsfunktionen
#######################

def neighborhood_density(x, y, agent_type, grid_objects):
    count = 0
    total = 0
    for dx in range(-RADIUS, RADIUS + 1):
        for dy in range(-RADIUS, RADIUS + 1):
            if abs(dx) + abs(dy) <= RADIUS:
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                total += 1
                if grid_objects[nx, ny] == agent_type:
                    count += 1
                elif agent_type == 3 and (grid_objects[nx, ny] == 1 or grid_objects[nx, ny] == 2):
                    count += 1
    return count / total if total > 0 else 0

def move_agent(agent, agent_id, grid_agents):
    directions = [(1,0), (-1,0), (0,1), (0,-1)]
    dx, dy = random.choice(directions)
    new_x = (agent.x + dx) % GRID_SIZE
    new_y = (agent.y + dy) % GRID_SIZE
    if grid_agents[new_x, new_y] == -1:
        grid_agents[agent.x, agent.y] = -1
        grid_agents[new_x, new_y] = agent_id
        agent.x = new_x
        agent.y = new_y

def pickup_or_drop(agent, grid_objects):
    f = neighborhood_density(agent.x, agent.y, agent.agent_type, grid_objects)
    cell_object = grid_objects[agent.x, agent.y]

    allowed = False
    if not agent.carrying and cell_object != 0:
        if agent.agent_type == 1 and cell_object == 1:
            allowed = True
        elif agent.agent_type == 2 and cell_object == 2:
            allowed = True
        elif agent.agent_type == 3:
            allowed = True

    if allowed:
        p_pick = (K_PLUS / (K_PLUS + f)) ** 2
        if random.random() < p_pick:
            grid_objects[agent.x, agent.y] = 0
            agent.carrying = cell_object

    elif agent.carrying and grid_objects[agent.x, agent.y] == 0:
        p_drop = (f / (K_MINUS + f)) ** 2
        if random.random() < p_drop:
            grid_objects[agent.x, agent.y] = agent.carrying
            agent.carrying = 0

def fmt(value, decimals=3):
    return str(round(value, decimals)).replace('.', ',')

def visualize(step, param_id, run, grid_objects):
    plt.clf()
    plt.title(f"Params {param_id+1} | Run {run+1} | Step {step}")
    cmap = ListedColormap(['white', 'red', 'blue'])
    plt.imshow(grid_objects.T, cmap=cmap, vmin=0, vmax=2)
    filename = f"2Obj3Ag_6_p{param_id+1}_A_{NUM_AGENTS_A}__B_{NUM_AGENTS_B}__AB_{NUM_AGENTS_AB}_r{run+1}.png"
    plt.savefig(filename, dpi=150)
    plt.close()



# csv
##########################

csv_filename = "sorting_simulation_results_6.csv"
fieldnames = [
    "param_id", "run", "step",
    "NUM_AGENTS_A", "NUM_AGENTS_B", "NUM_AGENTS_AB",
    "NUM_OBJECTS_A", "NUM_OBJECTS_B",
    "total_clusters", "red_clusters", "blue_clusters",
    "pure_red_cluster", "pure_blue_cluster",
    "Avg_weighted_Distance_red", "Avg_weighted_Distance_blue", "Avg_difference",
    "Avg_weighted_color_ratio", "mostly_pure"
]

with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
    writer.writeheader()


# Simulation
##########################
plt.ion()

for param_id, params in enumerate(parameter_sets):

    NUM_AGENTS_A  = params["NUM_AGENTS_A"]
    NUM_AGENTS_B  = params["NUM_AGENTS_B"]
    NUM_AGENTS_AB = params["NUM_AGENTS_AB"]
    NUM_OBJECTS_A = params["NUM_OBJECTS_A"]
    NUM_OBJECTS_B = params["NUM_OBJECTS_B"]
    NUM_AGENTS_List = [NUM_AGENTS_A, NUM_AGENTS_B, NUM_AGENTS_AB]
    STEPS = params["STEPS"]
    VISUALIZE_EVERY = STEPS-1
    

    aggregated = {}

    for run in range(RUNS_PER_PARAM):

        print(f"\n=== Params {param_id+1}/{len(parameter_sets)} | Run {run+1}/{RUNS_PER_PARAM} ===")
        print(f"    {params}")

        # --- Reset ---
        grid_objects = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        grid_agents  = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)
        agents = []
        agent_id = 0

        # Alle Objekte zusammen zufällig platzieren
        all_objects = [1] * NUM_OBJECTS_A + [2] * NUM_OBJECTS_B
        random.shuffle(all_objects)

        placed = 0
        while placed < len(all_objects):
            x = random.randrange(GRID_SIZE)
            y = random.randrange(GRID_SIZE)
            if grid_objects[x, y] == 0:
                grid_objects[x, y] = all_objects[placed]
                placed += 1

        agent_t = 0
        for NUM_AGENTS in NUM_AGENTS_List:
            agent_t += 1
            for i in range(NUM_AGENTS):
                while True:
                    x = random.randrange(GRID_SIZE)
                    y = random.randrange(GRID_SIZE)
                    if grid_agents[x, y] == -1:
                        grid_agents[x, y] = agent_id
                        agents.append(Agent(x, y, agent_t))
                        agent_id += 1
                        break

        for step in range(STEPS):
            for i, agent in enumerate(agents):
                move_agent(agent, i, grid_agents)
                pickup_or_drop(agent, grid_objects)

            if step % VISUALIZE_EVERY == 0 and step != 0:
                stats = cm.count_pure_clusters_and_color_ratios_2(2, GRID_SIZE, grid_objects, 12)

                total = stats.total
                pure_red_cluster = stats.pure_red_cluster
                pure_blue_cluster = stats.pure_blue_cluster
                avg_color_ratio = stats.avg_color_ratio
                avg_red = stats.avg_red
                avg_blue = stats.avg_blue
                avg_difference = stats.avg_difference
                mostly_pure = stats.mostly_pure

                red    = cm.count_clusters_type(2, 1, GRID_SIZE, grid_objects, 12)
                blue   = cm.count_clusters_type(2, 2, GRID_SIZE, grid_objects, 12)
                #avg_red, avg_blue, avg_difference = cm.cluster_compactness(2, GRID_SIZE, grid_objects, 15)

                print(f"  Step {step}: total={total}, red={red}, blue={blue}, "
                    f"compact_red={avg_red:.3f}, compact_blue={avg_blue:.3f}, avg_difference={avg_difference:.3f},"
                    f"pure_red={pure_red_cluster}, pure_blue={pure_blue_cluster}, "
                    f"avg_color_ratio={avg_color_ratio:.3f}")

                visualize(step, param_id, run, grid_objects)

                with open(csv_filename, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
                    writer.writerow({
                        "param_id":       param_id + 1,
                        "run":            run + 1,
                        "step":           step,
                        "NUM_AGENTS_A":   NUM_AGENTS_A,
                        "NUM_AGENTS_B":   NUM_AGENTS_B,
                        "NUM_AGENTS_AB":  NUM_AGENTS_AB,
                        "NUM_OBJECTS_A":  NUM_OBJECTS_A,
                        "NUM_OBJECTS_B":  NUM_OBJECTS_B,
                        "total_clusters": total,
                        "red_clusters":   red,
                        "blue_clusters":  blue,
                        "pure_red_cluster": pure_red_cluster,
                        "pure_blue_cluster": pure_blue_cluster,
                        "Avg_weighted_Distance_red":  fmt(avg_red),
                        "Avg_weighted_Distance_blue": fmt(avg_blue),
                        "Avg_difference" : fmt(avg_difference),
                        "Avg_weighted_color_ratio": fmt(avg_color_ratio), 
                        "mostly_pure" : mostly_pure,
                    })

                if step not in aggregated:
                    aggregated[step] = {
                        "total_clusters": [], "red_clusters": [], "blue_clusters": [], "pure_red_cluster": [], "pure_blue_cluster": [],
                        "Avg_weighted_Distance_red": [], "Avg_weighted_Distance_blue": [], "Avg_difference": [], "Avg_weighted_color_ratio": [], "mostly_pure": []
                    }
                aggregated[step]["total_clusters"].append(total)
                aggregated[step]["red_clusters"].append(red)
                aggregated[step]["blue_clusters"].append(blue)
                aggregated[step]["pure_red_cluster"].append(pure_red_cluster)
                aggregated[step]["pure_blue_cluster"].append(pure_blue_cluster)
                aggregated[step]["Avg_weighted_Distance_red"].append(avg_red)
                aggregated[step]["Avg_weighted_Distance_blue"].append(avg_blue)
                aggregated[step]["Avg_difference"].append(avg_difference)
                aggregated[step]["Avg_weighted_color_ratio"].append(avg_color_ratio)
                aggregated[step]["mostly_pure"].append(mostly_pure)

    #Durchschnitt schreiben
    with open(csv_filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')
        writer.writerow({})
        for step, vals in sorted(aggregated.items()):
            writer.writerow({
                "param_id":          f"AVG_p{param_id+1}",
                "run":               "AVG",
                "step":              step,
                "NUM_AGENTS_A":      NUM_AGENTS_A,
                "NUM_AGENTS_B":      NUM_AGENTS_B,
                "NUM_AGENTS_AB":     NUM_AGENTS_AB,
                "NUM_OBJECTS_A":     NUM_OBJECTS_A,
                "NUM_OBJECTS_B":     NUM_OBJECTS_B,
                "total_clusters":    round(np.mean(vals["total_clusters"]), 3),
                "red_clusters":      round(np.mean(vals["red_clusters"]), 3),
                "blue_clusters":     round(np.mean(vals["blue_clusters"]), 3),
                "pure_red_cluster":      round(np.mean(vals["pure_red_cluster"]), 3),
                "pure_blue_cluster":     round(np.mean(vals["pure_blue_cluster"]), 3),
                "Avg_weighted_Distance_red":  fmt(np.mean(vals["Avg_weighted_Distance_red"])),
                "Avg_weighted_Distance_blue": fmt(np.mean(vals["Avg_weighted_Distance_blue"])),
                "Avg_difference": fmt(np.mean(vals["Avg_difference"])),
                "Avg_weighted_color_ratio": fmt(np.mean(vals["Avg_weighted_color_ratio"])),
                "mostly_pure" : " "
            })
        writer.writerow({})

plt.ioff()
plt.show()