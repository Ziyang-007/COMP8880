import graph_tool.all as gt
import numpy as np

from tqdm import tqdm
import os

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

# 从 .gt 文件中加载图并显示进度条
graph = load_graph_with_progress("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/recommendation_network.gt")

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




# # 假设 graph 是已经创建的 Graph-tool 图对象
# degree_map = graph.degree_property_map("total")  # 获取每个节点的度数
# degrees = degree_map.a  # 度数数组

# # 找到度数最高的20个节点的索引
# top20_indices = np.argsort(-degrees)[:5000]  # 使用负号进行降序排列

# # 创建一个子图，仅包含度数最高的20个节点
# subgraph = gt.GraphView(graph, vfilt=lambda v: v in top20_indices)
# print("Finish generate sub-network")

# # 为子图计算布局
# pos_subgraph = gt.sfdp_layout(subgraph)

# # 绘制子图，可以调整节点大小和颜色以高亮显示这些重要的节点
# gt.graph_draw(subgraph, pos=pos_subgraph, vertex_size=gt.prop_to_size(degree_map, mi=2, ma=10),
#               vertex_fill_color='red', edge_pen_width=2,
#               output_size=(1000, 1000), output="top5000_high_degree_nodes.png")