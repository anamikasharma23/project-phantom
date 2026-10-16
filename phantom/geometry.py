"""Small geometry helpers shared by the environment and the agent.

Coordinates are (x, y) grid cells. x grows to the right, y grows DOWN
(like a screen). Bearings are compass style: 0 deg = north (up), 90 = east,
measured clockwise.
"""
import heapq
import math

import numpy as np

# Terrain codes used everywhere in the project.
PLAIN, ROUGH, ROCK = 0, 1, 2
TERRAIN_NAMES = {PLAIN: "plain", ROUGH: "rough", ROCK: "rock"}

# The four legal movement directions (4-connected grid keeps it simple).
DIRECTIONS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def bearing_deg(origin, target):
    """Compass bearing from origin to target, in degrees [0, 360)."""
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_diff_deg(a, b):
    """Smallest signed difference a - b, in degrees, range (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def move_cost(terrain_code, costs):
    """Battery cost of entering a cell. Rock is impassable (None)."""
    if terrain_code == ROCK:
        return None
    return costs["move_rough"] if terrain_code == ROUGH else costs["move_plain"]


def dijkstra(terrain, start, costs):
    """Cheapest battery cost from `start` to every cell (inf if unreachable).

    Returns (cost_grid, parent_dict). Used for path planning and for the
    "can I still get home?" safety check.
    """
    h, w = terrain.shape
    dist = np.full((h, w), np.inf)
    parent = {}
    sx, sy = start
    dist[sy, sx] = 0.0
    heap = [(0.0, sx, sy)]
    while heap:
        d, x, y = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            c = move_cost(terrain[ny, nx], costs)
            if c is None:
                continue
            nd = d + c
            if nd < dist[ny, nx] - 1e-12:
                dist[ny, nx] = nd
                parent[(nx, ny)] = (x, y)
                heapq.heappush(heap, (nd, nx, ny))
    return dist, parent


def path_from_parents(parent, start, goal):
    """Rebuild the cell path start -> goal (both included)."""
    if tuple(start) == tuple(goal):
        return [tuple(start)]
    if tuple(goal) not in parent:
        return None
    path = [tuple(goal)]
    while path[-1] != tuple(start):
        path.append(parent[path[-1]])
    path.reverse()
    return path


def direction_between(a, b):
    """Direction letter for a single step from cell a to neighbour b."""
    step = (b[0] - a[0], b[1] - a[1])
    for name, vec in DIRECTIONS.items():
        if vec == step:
            return name
    raise ValueError(f"cells {a} and {b} are not neighbours")
