import networkx as nx
from tqdm import tqdm
import pickle

def build_graph_from_file(file_path):
    # Create an empty graph object
    graph = nx.Graph()
    
    # Count the total number of lines to set up tqdm's progress bar
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    # Open the text file and read each line with a progress bar
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Building Network"):
            # Split the line by spaces
            nodes = line.strip().split()
            
            if len(nodes) > 4:
                # The first node is the main node
                main_node = nodes[0]
                # The subsequent nodes are the connected nodes
                connected_nodes = nodes[1:]
                
                # Add edges to the graph between the main node and each connected node
                for connected_node in connected_nodes:
                    graph.add_edge(main_node, connected_node)
    
    return graph


import matplotlib.pyplot as plt
def read_data_from_file(file_path):
    len_nodes_distribution = []
    # Count the total number of lines to set up tqdm's progress bar
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    # Open the text file and read each line with a progress bar
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Building Network"):
            # Split the line by spaces
            nodes = line.strip().split()
            len_nodes_distribution.append(len(nodes))

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(len_nodes_distribution, bins=range(1, max(len_nodes_distribution)+1), edgecolor='black', alpha=0.7)
    plt.title('Distribution of Number of Nodes per Line')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(len_nodes_distribution)+1))
    plt.grid(True)
    plt.show()
            

# Specify the path to your text file
file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/Code/recommendation_network.txt'

# read_data_from_file(file_path)  

# Build the network graph
network_graph = build_graph_from_file(file_path)

# Python Pickle format (binary)
# with open('network_graph_pruned.pkl', 'wb') as f:
#     pickle.dump(network_graph, f)

# Print information about the graph
print(f"Nodes: {len(network_graph.nodes())}")
print(f"Edges: {len(network_graph.edges())}")
