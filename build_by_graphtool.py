import graph_tool.all as gt
from tqdm import tqdm

def build_graph_from_file(file_path):
    # 创建一个无向图
    graph = gt.Graph(directed=False)
    # 创建一个字典以存储节点ID到图节点的映射
    node_map = {}

    # 计算文件中的行数
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)

    # 使用进度条读取文件
    with open(file_path, 'r') as file:
        for line in tqdm(file, total=total_lines, desc="Building Network"):
            nodes = line.strip().split()
            
            if len(nodes) > 160:
                # 确保主节点存在于图中
                main_node = nodes[0]
                if main_node not in node_map:
                    node_map[main_node] = graph.add_vertex()

                # 处理连接的节点
                connected_nodes = nodes[1:]
                for connected_node in connected_nodes:
                    if connected_node not in node_map:
                        node_map[connected_node] = graph.add_vertex()
                    # 在主节点和连接节点之间添加边
                    graph.add_edge(node_map[main_node], node_map[connected_node])
    
    return graph

file_path = '/home/fengziyang/ANU/COMP8880/recommendation_network.txt'
graph = build_graph_from_file(file_path)

num_vertices = graph.num_vertices()
num_edges = graph.num_edges()
print(f"Number of vertices: {num_vertices}")
print(f"Number of edges: {num_edges}")


# 随机选择一定比例的节点进行绘图
import numpy as np
subsample_size = int(0.1 * graph.num_vertices())  # 例如，只取10%
all_vertices = np.random.choice(graph.get_vertices(), subsample_size, replace=False)
subgraph = gt.GraphView(graph, vfilt=lambda v: v in all_vertices)
print("Finish generate sub-graph")
pos_subsample = gt.sfdp_layout(subgraph)
gt.graph_draw(subgraph, pos_subsample, vertex_size=1, edge_pen_width=0.1,
              output_size=(1000, 1000), output="graph_visualization_subsample.png")
