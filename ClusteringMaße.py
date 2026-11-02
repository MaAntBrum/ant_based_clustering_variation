
"""
ClusteringMaße.py

Effiziente Analyse eines zweidimensionalen, periodischen Gitters.

Objektwerte:
    0 = leer
    1 = rot
    2 = blau

Cluster werden über eine Manhattan-Nachbarschaft verbunden.
Ein Cluster wird bei Mindestgröße d nur berücksichtigt, wenn
seine Größe strikt größer als d ist
"""

from functools import lru_cache
from dataclasses import dataclass
import numpy as np



# Nachbarschaft
@lru_cache(maxsize=None)
def make_neighborhood(radius):
    """Berechnet die Manhattan-Nachbarschaft einmal pro Radius."""
    if radius < 0:
        raise ValueError("radius darf nicht negativ sein.")

    return tuple(
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    )



# Zentrale Clustererkennung
def find_clusters(grid_objects, radius=2):
    grid = np.asarray(grid_objects)

    if grid.ndim != 2:
        raise ValueError("grid_objects muss zweidimensional sein.")

    size_x, size_y = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    neighbors = make_neighborhood(radius)

    clusters = []

    for x in range(size_x):
        for y in range(size_y):

            if grid[x, y] == 0 or visited[x, y]:
                continue

            stack = [(x, y)]
            visited[x, y] = True
            cells = []

            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))

                for dx, dy in neighbors:
                    nx = (cx + dx) % size_x
                    ny = (cy + dy) % size_y

                    if (
                        grid[nx, ny] != 0
                        and not visited[nx, ny]
                    ):
                        # Bereits beim Einfügen markieren:
                        # verhindert doppelte Stack-Einträge.
                        visited[nx, ny] = True
                        stack.append((nx, ny))

            clusters.append(np.asarray(cells, dtype=np.intp))

    return clusters


def _get_large_clusters(
    NUMBER_OF_CLUSTER_RADIUS,
    grid_objects,
    d,
):
    return [
        cluster
        for cluster in find_clusters(
            grid_objects,
            NUMBER_OF_CLUSTER_RADIUS,
        )
        if len(cluster) > d
    ]

#zähle rote und blaue Zellen eines Clusters
def _cluster_color_counts(grid, cells):
    values = grid[cells[:, 0], cells[:, 1]]

    red_count = np.count_nonzero(values == 1)
    blue_count = np.count_nonzero(values == 2)

    return int(red_count), int(blue_count)


def spatial_entropy(s, GRID_SIZE, grid_objects):
    grid = np.asarray(grid_objects)
    size_x, size_y = grid.shape

    if s <= 0:
        raise ValueError("Die Patchgröße s muss positiv sein.")

    if s == size_x and s == size_y:
        p = np.count_nonzero(grid) / grid.size
        return float(-p * np.log(p)) if p > 0 else 0.0

    entropy = 0.0

    # Schneller, vektorisierter Weg für gleich große Patches.
    if size_x % s == 0 and size_y % s == 0:
        occupied = (grid != 0).reshape(
            size_x // s, s,
            size_y // s, s,
        )

        # Belegung pro Patch.
        counts = occupied.sum(axis=(1, 3))
        probabilities = counts / (s * s)

        valid = probabilities > 0

        return float(
            -np.sum(
                probabilities[valid]
                * np.log(probabilities[valid])
            )
        )

    # Allgemeiner Weg mit periodischen Randbedingungen.
    for x in range(0, size_x, s):
        for y in range(0, size_y, s):
            count = 0

            for dx in range(s):
                for dy in range(s):
                    nx = (x + dx) % size_x
                    ny = (y + dy) % size_y

                    if grid[nx, ny] != 0:
                        count += 1

            p = count / (s * s)

            if p > 0:
                entropy -= p * np.log(p)

    return float(entropy)



# ANZAHL ALLER CLUSTER
def count_clusters(
    NUMBER_OF_CLUSTER_RADIUS,
    GRID_SIZE,
    grid_objects,
    d,
):
    return len(
        _get_large_clusters(
            NUMBER_OF_CLUSTER_RADIUS,
            grid_objects,
            d,
        )
    )



# Anzahl farbspezifischer Objekte in Clustern
def count_clusters_type(
    NUMBER_OF_CLUSTER_RADIUS,
    obj_type,
    GRID_SIZE,
    grid_objects,
    d,
):
    grid = np.asarray(grid_objects)

    if obj_type not in (1, 2):
        raise ValueError("obj_type muss 1 (rot) oder 2 (blau) sein.")

    cluster_count = 0

    for cluster in find_clusters(
        grid,
        NUMBER_OF_CLUSTER_RADIUS,
    ):
        values = grid[cluster[:, 0], cluster[:, 1]]
        target_count = np.count_nonzero(values == obj_type)

        if target_count > d:
            cluster_count += 1

    return cluster_count



# PERIODISCHER CLUSTER-MITTELPUNKT -> berechnet den Cluster Mittelpunkt
def _circular_center(cells, grid_size):
    xs = cells[:, 0]
    ys = cells[:, 1]

    angles_x = 2 * np.pi * xs / grid_size
    angles_y = 2 * np.pi * ys / grid_size

    mean_x = np.arctan2(
        np.mean(np.sin(angles_x)),
        np.mean(np.cos(angles_x)),
    )

    mean_y = np.arctan2(
        np.mean(np.sin(angles_y)),
        np.mean(np.cos(angles_y)),
    )

    center_x = int(
        np.rint(mean_x * grid_size / (2 * np.pi))
    ) % grid_size

    center_y = int(
        np.rint(mean_y * grid_size / (2 * np.pi))
    ) % grid_size

    return center_x, center_y

#mittlere Abstand roter und blauer Objekte zum zirkulären Cluster-Mittelpunkt
def _color_distances(grid, cells, grid_size):
    center_x, center_y = _circular_center(cells, grid_size)

    xs = cells[:, 0]
    ys = cells[:, 1]

    dx = np.abs(xs - center_x)
    dy = np.abs(ys - center_y)

    dx = np.minimum(dx, grid_size - dx)
    dy = np.minimum(dy, grid_size - dy)

    distances = np.hypot(dx, dy)
    values = grid[xs, ys]

    red_distances = distances[values == 1]
    blue_distances = distances[values == 2]

    avg_red = (
        float(np.mean(red_distances))
        if red_distances.size
        else None
    )

    avg_blue = (
        float(np.mean(blue_distances))
        if blue_distances.size
        else None
    )

    return avg_red, avg_blue

#Berechnet die größen-gewichteten mittleren Abstände roter und blauer Objekte zum Cluster-Mittelpunkt. Berücksichtigt werden alle Cluster mit mehr als d Zellen.
def cluster_compactness(
    NUMBER_OF_CLUSTER_RADIUS,
    GRID_SIZE,
    grid_objects,
    d,
):
    grid = np.asarray(grid_objects)
    grid_size = grid.shape[0]

    red_means = []
    blue_means = []
    weights_red = []
    weights_blue = []

    for cluster in _get_large_clusters(
        NUMBER_OF_CLUSTER_RADIUS,
        grid,
        d,
    ):
        avg_red, avg_blue = _color_distances(
            grid,
            cluster,
            grid_size,
        )

        cluster_size = len(cluster)

        if avg_red is not None:
            red_means.append(avg_red)
            weights_red.append(cluster_size)

        if avg_blue is not None:
            blue_means.append(avg_blue)
            weights_blue.append(cluster_size)

    avg_red = (
        float(np.average(red_means, weights=weights_red))
        if red_means
        else 0.0
    )

    avg_blue = (
        float(np.average(blue_means, weights=weights_blue))
        if blue_means
        else 0.0
    )

    return avg_red, avg_blue, avg_blue - avg_red


#pure Cluster und Farbverhältnis

def _analyze_cluster_colors(
    NUMBER_OF_CLUSTER_RADIUS,
    GRID_SIZE,
    grid_objects,
    d,
    ratio_order="red_blue",
    calculate_compactness=False,
):
    grid = np.asarray(grid_objects)
    grid_size = grid.shape[0]

    cluster_count = 0
    pure_red_cluster = 0
    pure_blue_cluster = 0

    color_ratios = []
    color_ratio_weights = []

    red_distance_means = []
    blue_distance_means = []
    distance_weights = []

    for cluster in _get_large_clusters(
        NUMBER_OF_CLUSTER_RADIUS,
        grid,
        d,
    ):
        cluster_size = len(cluster)

        red_count, blue_count = _cluster_color_counts(
            grid,
            cluster,
        )

        cluster_count += 1

        if red_count <= 5:
            pure_blue_cluster += 1

        elif blue_count <= 5:
            pure_red_cluster += 1

        else:
            if ratio_order == "red_blue":
                ratio = red_count / blue_count
            else:
                ratio = blue_count / red_count

            color_ratios.append(ratio)
            color_ratio_weights.append(cluster_size)

            if calculate_compactness:
                avg_red, avg_blue = _color_distances(
                    grid,
                    cluster,
                    grid_size,
                )

                # Gemischte Cluster haben mindestens 6 Objekte
                # jeder Farbe; beide Mittelwerte existieren.
                red_distance_means.append(avg_red)
                blue_distance_means.append(avg_blue)
                distance_weights.append(cluster_size)

    avg_color_ratio = (
        float(
            np.average(
                color_ratios,
                weights=color_ratio_weights,
            )
        )
        if color_ratios
        else 0.0
    )

    if calculate_compactness:
        avg_red = (
            float(np.average(
                red_distance_means,
                weights=distance_weights,
            ))
            if red_distance_means
            else 0.0
        )

        avg_blue = (
            float(np.average(
                blue_distance_means,
                weights=distance_weights,
            ))
            if blue_distance_means
            else 0.0
        )

        avg_difference = avg_blue - avg_red

        mostly_pure = (
            pure_blue_cluster + pure_red_cluster
            > cluster_count / 2
        )

        return (
            cluster_count,
            pure_red_cluster,
            pure_blue_cluster,
            avg_color_ratio,
            avg_red,
            avg_blue,
            avg_difference,
            mostly_pure,
        )

    return (
        cluster_count,
        pure_red_cluster,
        pure_blue_cluster,
        avg_color_ratio,
    )

#zählt reine Cluster und berechnet das größen-gewichtete Verhältnis ROT/BLAU für gemischte Cluster
def count_pure_clusters_and_color_ratios(
    NUMBER_OF_CLUSTER_RADIUS,
    GRID_SIZE,
    grid_objects,
    d,
):
    return _analyze_cluster_colors(
        NUMBER_OF_CLUSTER_RADIUS,
        GRID_SIZE,
        grid_objects,
        d,
        ratio_order="red_blue",
        calculate_compactness=False,
    )



@dataclass
class ClusterStatistics:
    total: int
    pure_red_cluster: int
    pure_blue_cluster: int
    avg_color_ratio: float
    avg_red: float
    avg_blue: float
    avg_difference: float
    mostly_pure: int


"""
Erweiterte Clusteranalyse mit Farbverhältnis und Kompaktheit.

Rückgabe:
    total_clusters,
    pure_red_cluster,
    pure_blue_cluster,
    avg_color_ratio,
    avg_red,
    avg_blue,
    avg_difference,
    mostly_pure
"""
def count_pure_clusters_and_color_ratios_2(
    NUMBER_OF_CLUSTER_RADIUS,
    GRID_SIZE,
    grid_objects,
    d,
) -> ClusterStatistics:

    result = _analyze_cluster_colors(
        NUMBER_OF_CLUSTER_RADIUS,
        GRID_SIZE,
        grid_objects,
        d,
        ratio_order="red_blue",
        calculate_compactness=True,
    )

    return ClusterStatistics(*result)
