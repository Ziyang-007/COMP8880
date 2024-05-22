import graph_tool.all as gt
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def load_graph_with_progress(file_path):
    # 获取文件大小
    file_size = os.path.getsize(file_path)
    
    # 初始化进度条
    pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Loading graph")
    
    # 使用 graph_tool 加载图
    graph = gt.load_graph(file_path)
    
    # 更新进度条以反映加载完成
    pbar.update(file_size)
    pbar.close()
    
    return graph

def analyze_and_draw(graph):
    # 打印图的信息以验证加载成功
    print("Number of vertices in loaded graph:", graph.num_vertices())
    print("Number of edges in loaded graph:", graph.num_edges())

    # 度中心性：节点的度（即连接到节点的边数）是网络分析中最基本的度量之一。
    degree_map = graph.degree_property_map("total")  # 'out', 'in', 'total'
    degrees = degree_map.a  # 获取度数组
    print("Degrees of the vertices:", degrees)

    # 全局聚类系数：衡量节点间三角关系闭合的倾向，反映了网络的聚集程度。
    clust_global = gt.global_clustering(graph)
    print("Global clustering coefficient:", clust_global[0])

    # 最大连通分量：找出图中最大的连通子图。
    largest_comp = gt.label_largest_component(graph)
    lcc_graph = gt.GraphView(graph, vfilt=largest_comp)
    print("Number of vertices in largest component:", lcc_graph.num_vertices())



    # # 选取子图节点数
    # N = 50000
    # # 假设 graph 是已经创建的 Graph-tool 图对象
    # degree_map = graph.degree_property_map("total")  # 获取每个节点的度数
    # degrees = degree_map.a  # 度数数组

    # # 找到度数最高的N个节点的索引
    # topN_indices = np.argsort(-degrees)[:N]  # 使用负号进行降序排列

    # # 创建一个子图，仅包含度数最高的N个节点
    # topN_set = set(topN_indices)
    # subgraph = gt.GraphView(graph, vfilt=lambda v: v in topN_set)
    # print("Finish generate sub-network")

    # # 为子图计算布局
    # # pos_subgraph = gt.fruchterman_reingold_layout(subgraph)
    # pos_subgraph = gt.sfdp_layout(subgraph)
    # print("Finish generate layout")

    # # 绘制子图，可以调整节点大小和颜色以高亮显示这些重要的节点
    # gt.graph_draw(subgraph, pos=pos_subgraph, vertex_size=gt.prop_to_size(degree_map, mi=5, ma=20),
    #               output_size=(3000, 3000), output=f"test_top{N}_network_nodes.png")




    # 假设 graph 是已经创建的 Graph-tool 图对象
    # 获取最大连通组件的过滤器
    largest_comp = gt.label_largest_component(graph)
    # 创建一个只包含最大连通组件的子图
    lcc_graph = gt.GraphView(graph, vfilt=largest_comp)
    print("Finish generate sub-network of the largest connected component")

    # 计算最大连通子图中所有节点的度数
    degree_map_lcc = lcc_graph.degree_property_map("total")
    degrees_lcc = degree_map_lcc.a  # 度数数组

    # 选取度数最高的10000个节点
    N = 260000
    topN_indices = np.argsort(-degrees_lcc)[:N]  # 使用负号进行降序排列

    # 创建一个子图，仅包含度数最高的N个节点
    topN_set = set(topN_indices)
    topN_subgraph = gt.GraphView(lcc_graph, vfilt=lambda v: v in topN_set)
    print(f"Finish generate sub-network of top {N} nodes in the largest component")

    # 为子图计算布局
    pos_topN_subgraph = gt.sfdp_layout(topN_subgraph)
    print(f"Finish generate layout for top {N} nodes")

    # 绘制子图，可以调整节点大小和颜色以高亮显示这些重要的节点
    gt.graph_draw(topN_subgraph, pos=pos_topN_subgraph, vertex_size=gt.prop_to_size(degree_map_lcc, mi=5, ma=20),
                output_size=(8000, 8000), output=f"test_top{N}_nodes_in_largest_cc.png")
    
def get_connected_nodes(graph, node_map, node_id):
    # Get the vertex index from the node map
    vertex_index = node_map.get(node_id)
    if vertex_index is None:
        return None  # Node ID not found

    # Get the vertex from the graph
    vertex = graph.vertex(vertex_index)
    
    # Retrieve connected vertices and convert their indices to string IDs
    connected_vertices = [graph.vertex_index[v] for v in vertex.all_neighbors()]
    connected_vertices_ids = [key for key, value in node_map.items() if value in connected_vertices]

    return connected_vertices_ids

def find_shortest_path(graph, node_map, node_id1, node_id2):
    # Get the vertex indices for the two nodes
    vertex_index1 = node_map.get(node_id1)
    vertex_index2 = node_map.get(node_id2)

    if vertex_index1 is None or vertex_index2 is None:
        return None, "One or both node IDs not found."

    # Get the vertices from the graph
    vertex1 = graph.vertex(vertex_index1)
    vertex2 = graph.vertex(vertex_index2)
    
    # Find the shortest path between the vertices
    path_vertices, path_edges = gt.shortest_path(graph, vertex1, vertex2)

    # If there are vertices in the path, map their indices back to string IDs
    if path_vertices:
        path_vertex_ids = [key for key, value in node_map.items() if value in [int(v) for v in path_vertices]]
    else:
        path_vertex_ids = None

    path_length = len(path_edges) if path_edges else 0

    return path_vertex_ids, path_length

def evaluation_nlp_score(graph, node_map, test_node_id, sim_node_id):
    path, length = find_shortest_path(graph, node_map, test_node_id, sim_node_id)
    if path == None:
        score = 0
    else:
        score = 1 * (0.9 ** (length - 1))
    return score

def node_distance_expectation(graph, distance_threshold=310000):
    # 获取图中的所有节点
    nodes = list(graph.vertices())
    
    # 用于存储每个节点到其他所有节点的平均最短距离
    average_distances = []
    
    # 使用 tqdm 显示进度条
    for node in tqdm(nodes, desc="Calculating shortest distances"):
        # 获取该节点到所有其他节点的最短距离
        distances = gt.shortest_distance(graph, source=node).get_array()
        
        # 过滤掉距离为无限大或者大于阈值的情况
        valid_distances = [d for d in distances if d < distance_threshold]
        
        if len(valid_distances) > 0:
            # 计算该节点到所有其他节点的平均最短距离
            avg_distance = sum(valid_distances) / len(valid_distances)
            average_distances.append(avg_distance)
        print(sum(average_distances) / len(average_distances))
    
    # 计算所有节点的平均最短距离的平均值
    if len(average_distances) > 0:
        total_average_distance = sum(average_distances) / len(average_distances)
    else:
        total_average_distance = float('inf')
    
    return total_average_distance

def degree_distribution(graph):
    degrees = graph.get_out_degrees(graph.get_vertices())

    degrees = degrees.astype(int)
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 1), edgecolor='black')
    plt.title('Degree Distribution')
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.yscale('log')  # Optional: to better visualize the distribution
    plt.grid(True)
    plt.show()
    
def nlp_score_diagram(sorted_scores):

    sorted_scores = sorted(total_score, reverse=True)
    plt.figure(figsize=(10, 6))

    from scipy.stats import gaussian_kde

    kde = gaussian_kde(sorted_scores, bw_method=0.1)
    density = kde(sorted_scores)

    norm_density = (density - min(density)) / (max(density) - min(density))

    sc = plt.scatter(range(len(sorted_scores)), sorted_scores, c=norm_density, cmap='viridis')
    plt.colorbar(sc, label='Density')

    plt.title('Score Distribution')
    plt.ylabel('Score')
    plt.grid(True)
    plt.xticks([])  # Hide x-axis labels

    plt.show()

def process_line(graph_path, node_map_path, line):
    one_node_score = []
    nodes = line.strip().split()
    test_node = nodes[0]
    for sim_node in nodes[1:]:
        score = evaluation_nlp_score(graph_path, node_map_path, test_node, sim_node)
        one_node_score.append(score)
    return sum(one_node_score) / len(one_node_score) if one_node_score else 0


if __name__ == '__main__':
    # 从 .gt 文件中加载图并显示进度条, 从 .pkl中加载id映射
    graph = gt.load_graph("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/50_recommendation_network.gt")
    dict_path = "/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/50_node_map.pkl"
    with open(dict_path, "rb") as f:
        node_map = pickle.load(f)
        
    # degree_distribution(graph)
    
    # NLP评估
    sim_node_file = "/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/similar_products_data_50_product_meta_more_infos.txt"
    total_score = []
    num_lines = sum(1 for _ in open(sim_node_file, 'r'))

    with open(sim_node_file, "r") as recommendation_node:
        lines = recommendation_node.readlines()
    
    # 使用进程池进行并行处理
    with ProcessPoolExecutor(max_workers=11) as executor:  # Adjust the number of processes as needed
        futures = [executor.submit(process_line, graph, node_map, line) for line in lines]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing lines"):
            total_score.append(future.result())
    
    print("Finish generate total_score list")
    nlp_score_diagram(total_score)
    
    print(sum(total_score)/len(total_score))

    # 分析网络和画图
    # analyze_and_draw(graph)
