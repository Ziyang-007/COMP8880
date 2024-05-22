import graph_tool.all as gt

# 读取网络文件
graph = gt.load_graph("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/recommendation_network.gt")
print("Finish load model")

# # 使用社区检测算法
# state = gt.minimize_blockmodel_dl(graph)

# # 获取社区数量
# num_communities = state.get_B()

# print(f"社区数量: {num_communities}")

# 全局聚类系数：衡量节点间三角关系闭合的倾向，反映了网络的聚集程度。
clust_global = gt.global_clustering(graph)
print("Global clustering coefficient:", clust_global[0])

# 计算每个节点的度数
degree_property_map = graph.degree_property_map("total")

# 计算网络的平均度数
average_degree = sum(degree_property_map.a) / graph.num_vertices()

print(f"网络的平均度数: {average_degree:.2f}")
