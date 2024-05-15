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
            
            if len(nodes) > 0:
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

file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/recommendation_network_node.txt'
graph = build_graph_from_file(file_path)

num_vertices = graph.num_vertices()
num_edges = graph.num_edges()
print(f"Number of vertices: {num_vertices}")
print(f"Number of edges: {num_edges}")

degree_map = graph.degree_property_map("total")  # 'out', 'in', 'total'
degrees = degree_map.a  # 获取度数组
print("Degrees of the vertices:", degrees)

largest_comp = gt.label_largest_component(graph)
lcc_graph = gt.GraphView(graph, vfilt=largest_comp)
print("Number of vertices in largest component:", lcc_graph.num_vertices())

graph.save("recommendation_network.gt")

# # 接近中心性
# closeness = gt.closeness(graph)
# print("Closeness centrality:", closeness.a)

# # 中介中心性
# betweenness = gt.betweenness(graph)[0]
# print("Betweenness centrality:", betweenness.a)


# pos = gt.sfdp_layout(graph)
# gt.graph_draw(graph, pos, vertex_size=gt.prop_to_size(degree_map, mi=5, ma=15),
#               vertex_fill_color=blocks, vertex_text=graph.vertex_index,
#               output_size=(1000, 1000), output="graph_visualization.png")

# import graph_tool.all as gt
# import numpy as np

# # 假设 graph 是已经创建的 Graph-tool 图对象
# degree_map = graph.degree_property_map("total")  # 获取每个节点的度数
# degrees = degree_map.a  # 度数数组

# # 找到度数最高的20个节点的索引
# top20_indices = np.argsort(-degrees)[:500]  # 使用负号进行降序排列

# # 创建一个子图，仅包含度数最高的20个节点
# subgraph = gt.GraphView(graph, vfilt=lambda v: v in top20_indices)
# print("Finish generate sub-network")

# # 为子图计算布局
# pos_subgraph = gt.sfdp_layout(subgraph)

# # 绘制子图，可以调整节点大小和颜色以高亮显示这些重要的节点
# gt.graph_draw(subgraph, pos=pos_subgraph, vertex_size=gt.prop_to_size(degree_map, mi=2, ma=10),
#               vertex_fill_color='red', edge_pen_width=2,
#               output_size=(1000, 1000), output="top500_high_degree_nodes.png")