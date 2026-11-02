import numpy as np                      #für unser 2D-Array
import random
import matplotlib.pyplot as plt         #Plot-Modul für Visualisierung
import ClusteringMaße as cm


# PARAMETER
#########################

GRID_SIZE = 50
NUM_AGENTS = 25
NUM_OBJECTS = 100
STEPS = 200001

RADIUS = 2
K_PLUS = 0.05
K_MINUS = 0.3

VISUALIZE_EVERY = 5000

# Agentenklasse  
class Agent:
    def __init__(self, x, y):             #Konsturktor zum erzeugen eines Agentens
        self.x = x
        self.y = y
        self.carrying = False


# init
#######################

grid_objects = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)      # 2D-Array: 0=leer, 1=Objekt
grid_agents = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)    # 2D-Array: -1=kein Agent, sonst = Agnet-ID ->sorgt für max 1 Agent pro Zelle

agents = []                                  #Liste alles Agent-Objekte

# Place objects randomly
placed = 0
while placed < NUM_OBJECTS:
    x = random.randrange(GRID_SIZE)
    y = random.randrange(GRID_SIZE)
    if grid_objects[x, y] == 0:             #Überprüft, ob sich dort bereits ein Objekt befindet
        grid_objects[x, y] = 1
        placed += 1

# Place agents randomly
for i in range(NUM_AGENTS):
    while True:                             #solange wiederholen, bis freie Zeile gefunden wurde
        x = random.randrange(GRID_SIZE)
        y = random.randrange(GRID_SIZE)
        if grid_agents[x, y] == -1:         #Testet, ob dort bereits ein Agent ist
            grid_agents[x, y] = i
            agents.append(Agent(x, y))      #Agenten zur Agentenliste hinzufügen
            break



# Hilfsfunktionen
########################

def neighborhood_density(x, y):             #berechnet die Dichte um (x,y)
    count = 0
    total = 0

    for dx in range(-RADIUS, RADIUS + 1):       #horizontale Überprüfung
        for dy in range(-RADIUS, RADIUS + 1):   #vertikale Überprüfung
            if abs(dx) + abs(dy) <= RADIUS:     #Manhattan-Distanz |dx|+|dy| ->Diamant Nachbarschaft anstatt Quadrat
                nx = (x + dx) % GRID_SIZE       # % GRID_SIZE, um die Torus-Randbedingung zu gewährleisten 
                ny = (y + dy) % GRID_SIZE
                total += 1                      #Anz Zellen +1
                if grid_objects[nx, ny] == 1:
                    count += 1                  #Anz Objekte +1

    return count / total if total > 0 else 0    #berechnet die Dichte f(x,y)=Anz Objekte in Nachbarschaft/Anz Zellen in Nachbarschaft


def move_agent(agent, agent_id):
    directions = [(1,0), (-1,0), (0,1), (0,-1)] #mögliche Bewegungen (4er-Nachbarschaft, nicht diagnonal)
    dx, dy = random.choice(directions)

    new_x = (agent.x + dx) % GRID_SIZE          #Neue Position mit Torus Eigenschaft
    new_y = (agent.y + dy) % GRID_SIZE

    if grid_agents[new_x, new_y] == -1:         #überprüft, ob neue Position frei ist
        grid_agents[agent.x, agent.y] = -1      #alte Position leeren
        grid_agents[new_x, new_y] = agent_id    #Agent auf neue Position setzen
        agent.x = new_x
        agent.y = new_y


def pickup_or_drop(agent):
    f = neighborhood_density(agent.x, agent.y)  #lokale Dichte berechnen

    if not agent.carrying and grid_objects[agent.x, agent.y] == 1:  #Agent trägt kein Objekt und Zelle hat Objekt
        p_pick = (K_PLUS / (K_PLUS + f)) ** 2                       #Berechnugn der Pick-Up Wahrscheinlichkeit
        if random.random() < p_pick:                                #Zufallsentscheidung (random.random() wählt Zahl zwischen 1 und 0)
            grid_objects[agent.x, agent.y] = 0
            agent.carrying = True

    elif agent.carrying and grid_objects[agent.x, agent.y] == 0:    #Agent trägt Objekt und Zelle ist frei
        p_drop = (f / (K_MINUS + f)) ** 2
        if random.random() < p_drop:
            grid_objects[agent.x, agent.y] = 1
            agent.carrying = False


def visualize(step):
    plt.clf()                                       #Löscht das Bild
    plt.imshow(grid_objects.T, cmap='gray')         #Zeigt Gitter als Bild
    plt.title(f"Step {step}")                       #Angezeigter Schritt

    filename = f"frame_{step}.png"
    plt.savefig(filename, dpi=150)
    plt.pause(0.01)



# Simulation Loop
########################################

plt.ion()                                       #Interaktiver Modus für Live Update

for step in range(STEPS):                       #Hauptschleife für die ausgeführten Schritte
    for i, agent in enumerate(agents):          #enumerate erzeugt (index,element) Paare, damit wir einen Index für das move_agent bekommen
        move_agent(agent, i)                    #Erst bewegen
        pickup_or_drop(agent)                   #dann Pick/Drop

    if step % VISUALIZE_EVERY == 0:             #Bild Updaten alle VISUALIZE_EVERY Schritte
            total_clusters_2 = cm.count_clusters(2, GRID_SIZE, grid_objects, 5)
            E5 = cm.spatial_entropy(5, GRID_SIZE, grid_objects)
            print(f"Step {step}: N5={total_clusters_2}, E5={E5:.3f}")
            visualize(step)


plt.ioff()                                      #interaktiven Modus beenden
plt.show()