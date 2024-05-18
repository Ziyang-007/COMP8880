import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
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
            
            if len(nodes) > 40:
                # The first node is the main node
                main_node = nodes[0]
                # The subsequent nodes are the connected nodes
                connected_nodes = nodes[1:]
                
                # Add edges to the graph between the main node and each connected node
                for connected_node in connected_nodes:
                    graph.add_edge(main_node, connected_node)
    
    return graph


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
    plt.title('Distribution of Connected Products')
    plt.xlabel('Number of Connected Products')
    plt.ylabel('Number of Products')
    # plt.xticks(range(1, max(len_nodes_distribution)+1))
    plt.grid(True)
    plt.show()
    
def analyze_graph(graph):
    # Check if the graph is connected
    if nx.is_connected(graph):
        diameter = nx.diameter(graph)
        average_path_length = nx.average_shortest_path_length(graph)
        clustering_coefficient = nx.average_clustering(graph)
        
        print(f"Network Diameter: {diameter}")
        print(f"Average Path Length: {average_path_length}")
        print(f"Clustering Coefficient: {clustering_coefficient}")
    else:
        print("Graph is not connected. Computing metrics for the largest connected component.")
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        
        diameter = nx.diameter(subgraph)
        average_path_length = nx.average_shortest_path_length(subgraph)
        clustering_coefficient = nx.average_clustering(subgraph)
        
        print(f"Network Diameter of the largest component: {diameter}")
        print(f"Average Path Length of the largest component: {average_path_length}")
        print(f"Clustering Coefficient of the largest component: {clustering_coefficient}")

def build_meta_node_list(file_path):
    # Count the total number of lines to set up tqdm's progress bar
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
    
    # Open the text file and read each line with a progress bar
    with open(file_path, 'r') as file:
        id_set = set()
        for line in tqdm(file, total=total_lines, desc="Building Network"):
            # Split the line by spaces
            nodes = line.strip().split()
            if nodes :
                for id in nodes:
                    id_set.add(id)
    
    return id_set

# Specify the path to your text file
file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/50_degree_nodes.txt'

# read_data_from_file(file_path)  

id_set = build_meta_node_list(file_path)
print(len(id_set))
with open("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/node_id.txt", 'w') as file:
    for id in id_set:
        file.write(id + "\n")


# Build the network graph
# network_graph = build_graph_from_file(file_path)

# Python Pickle format (binary)
# with open('network_graph_pruned.pkl', 'wb') as f:
#     pickle.dump(network_graph, f)

# # Print information about the graph
# print(f"Nodes: {len(network_graph.nodes())}")
# print(f"Edges: {len(network_graph.edges())}")

# # Analyze the pruned graph
# analyze_graph(network_graph)
