import graph_tool.all as gt

# 读取网络文件
graph = gt.load_graph("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/test_recommendation_network.gt")
print("Finish load model")

# 使用社区检测算法
state = gt.minimize_blockmodel_dl(graph)

# 获取社区数量
num_communities = state.get_B()

print(f"社区数量: {num_communities}")
