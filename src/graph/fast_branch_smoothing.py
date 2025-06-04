import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
import networkx as nx

def smooth_graph_branches_fast(G, smoothing_factor=0.1, num_points=None):
    """
    Fast spline smoothing for all branches in a graph.
    
    Args:
        G: NetworkX graph with 'pos' attributes
        smoothing_factor: Lower = smoother (0.0 to 1.0)
        num_points: Points per smoothed branch (None = auto)
    
    Returns:
        Dict of smoothed branches: {branch_id: (x_smooth, y_smooth)}
    """
    
    # 1. Find all branches (paths between degree != 2 nodes)
    branches = extract_branches_fast(G)
    
    # 2. Smooth each branch
    smoothed_branches = {}
    
    for i, branch in enumerate(branches):
        if len(branch) < 3:  # Need at least 3 points for spline
            # Just return original points
            coords = np.array([G.nodes[node]['pos'] for node in branch])
            smoothed_branches[i] = (coords[:, 0], coords[:, 1])
            continue
            
        # Get coordinates
        coords = np.array([G.nodes[node]['pos'] for node in branch])
        x, y = coords[:, 0], coords[:, 1]
        
        # Auto-determine number of output points
        if num_points is None:
            branch_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            n_out = max(len(branch), int(branch_length / 2))  # Adaptive
        else:
            n_out = num_points
            
        try:
            # Parametric spline fitting - FAST!
            tck, _ = splprep([x, y], s=smoothing_factor * len(branch), k=min(3, len(branch)-1))
            
            # Generate smooth curve
            u_new = np.linspace(0, 1, n_out)
            x_smooth, y_smooth = splev(u_new, tck)
            
            smoothed_branches[i] = (x_smooth, y_smooth)
            
        except Exception:
            # Fallback to original if spline fails
            smoothed_branches[i] = (x, y)
    
    return smoothed_branches

def extract_branches_fast(G):
    """
    Quickly extract all branches from graph.
    A branch is a path between nodes with degree != 2.
    """
    branches = []
    visited_edges = set()
    
    # Find all nodes that are not degree-2 (endpoints/junctions)
    special_nodes = [n for n in G.nodes if G.degree[n] != 2]
    
    for node in special_nodes:
        for neighbor in G.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge in visited_edges:
                continue
                
            # Trace branch from this edge
            branch = trace_branch(G, node, neighbor, visited_edges)
            if len(branch) > 1:
                branches.append(branch)
    
    return branches

def trace_branch(G, start, next_node, visited_edges):
    """Trace a branch until hitting a non-degree-2 node."""
    branch = [start, next_node]
    visited_edges.add(tuple(sorted([start, next_node])))
    
    current = next_node
    prev = start
    
    while G.degree[current] == 2:
        # Find the other neighbor
        neighbors = list(G.neighbors(current))
        next_node = neighbors[0] if neighbors[1] == prev else neighbors[1]
        
        edge = tuple(sorted([current, next_node]))
        if edge in visited_edges:
            break
            
        visited_edges.add(edge)
        branch.append(next_node)
        prev, current = current, next_node
    
    return branch

# Alternative: Even faster with simple moving average (if splines are too slow)
def smooth_branches_moving_average(G, window_size=3):
    """Ultra-fast smoothing using moving average."""
    branches = extract_branches_fast(G)
    smoothed_branches = {}
    
    for i, branch in enumerate(branches):
        if len(branch) < window_size:
            coords = np.array([G.nodes[node]['pos'] for node in branch])
            smoothed_branches[i] = (coords[:, 0], coords[:, 1])
            continue
            
        coords = np.array([G.nodes[node]['pos'] for node in branch])
        x, y = coords[:, 0], coords[:, 1]
        
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        x_smooth = np.convolve(x, kernel, mode='same')
        y_smooth = np.convolve(y, kernel, mode='same')
        
        # Keep endpoints unchanged
        x_smooth[0] = x[0]
        x_smooth[-1] = x[-1]
        y_smooth[0] = y[0]
        y_smooth[-1] = y[-1]
        
        smoothed_branches[i] = (x_smooth, y_smooth)
    
    return smoothed_branches

# Usage example:
def apply_smoothing(G, method='spline'):
    """
    Apply smoothing to your graph.
    
    Args:
        G: Your NetworkX graph
        method: 'spline' (slower, better) or 'moving_average' (faster)
    """
    if method == 'spline':
        return smooth_graph_branches_fast(G, smoothing_factor=0.05)
    else:
        return smooth_branches_moving_average(G, window_size=5)

# For visualization
def plot_smoothed_graph(G, smoothed_branches):
    """Quick plotting of smoothed results."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 8))
    
    # Plot original
    pos = nx.get_node_attributes(G, 'pos')
    x_orig = [pos[n][0] for n in G.nodes()]
    y_orig = [pos[n][1] for n in G.nodes()]
    plt.plot(x_orig, y_orig, 'r.-', alpha=0.3, label='Original', markersize=2)
    
    # Plot smoothed
    for branch_id, (x_smooth, y_smooth) in smoothed_branches.items():
        plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, alpha=0.8)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title('Original vs Smoothed Graph')
    plt.show()