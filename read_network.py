import networkx as nx
import pickle

def load_graph_from_pickle(file_path):
    # Open the pickle file and load the graph
    with open(file_path, 'rb') as f:
        graph = pickle.load(f)
    
    return graph

# Specify the path to your pickle file
file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/Code/network_graph.pkl'

# Load the graph from the pickle file
network_graph = load_graph_from_pickle(file_path)

# Print information about the graph
print(f"Nodes: {len(network_graph.nodes())}")
print(f"Edges: {len(network_graph.edges())}")