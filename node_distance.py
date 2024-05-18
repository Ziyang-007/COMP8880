import graph_tool.all as gt
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle

def calculate_avg_distance(graph_path, node_index, distance_threshold):
    graph = gt.load_graph(graph_path)
    node = graph.vertex(node_index)
    # 获取该节点到所有其他节点的最短距离
    distances = gt.shortest_distance(graph, source=node).get_array()

    # 过滤掉距离为无限大或者大于阈值的情况
    valid_distances = [d for d in distances if d < distance_threshold]

    if len(valid_distances) > 0:
        # 计算该节点到所有其他节点的平均最短距离
        avg_distance = sum(valid_distances) / len(valid_distances)
        return avg_distance
    else:
        return None

def node_distance_expectation(graph_path, distance_threshold=310000, num_processes=8):
    graph = gt.load_graph(graph_path)
    # 获取图中的所有节点索引
    node_indices = [int(node) for node in graph.vertices()]
    
    # 用于存储每个节点到其他所有节点的平均最短距离
    average_distances = []
    
    # 创建一个进程池
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # 提交所有任务
        futures = {executor.submit(calculate_avg_distance, graph_path, node_index, distance_threshold): node_index for node_index in node_indices}

        # 使用 tqdm 显示进度条
        for future in tqdm(as_completed(futures), total=len(futures), desc="Calculating shortest distances"):
            result = future.result()
            if result is not None:
                average_distances.append(result)
            # 打印当前所有节点的平均最短距离的平均值
            if average_distances:
                print(sum(average_distances) / len(average_distances))
    
    # 计算所有节点的平均最短距离的平均值
    if len(average_distances) > 0:
        total_average_distance = sum(average_distances) / len(average_distances)
    else:
        total_average_distance = float('inf')
    
    return total_average_distance

if __name__ == '__main__':
    # 从 .gt 文件中加载图并显示进度条, 从 .pkl中加载id映射
    graph_path = "/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/recommendation_network.gt"
    dict_path = "/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/network/node_map.pkl"
    with open(dict_path, "rb") as f:
        node_map = pickle.load(f)

    print(node_distance_expectation(graph_path, num_processes=8))  # Specify the number of processes here
