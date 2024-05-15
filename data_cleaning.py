import jsonlines
import json
from tqdm import tqdm
# 第一次清洗数据，获取所有的also buy和also view
# # Replace this with the actual file path
# input_file_path = '/Volumes/970EVOPLUS/AmazonReviewDataset/All_Amazon_Meta.json'

# # Path to your output text file
# output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta.txt'

def get_also_view_and_buy(input_file_path, output_file_path):
    # Count the number of lines for the progress bar (optional but helpful)
    num_lines = sum(1 for _ in open(input_file_path, 'r'))
    print("num lines: " + str(num_lines))

    # Open the input JSON lines file for reading and the output text file for writing in append mode
    with jsonlines.open(input_file_path) as reader, open(output_file_path, 'a') as output_file:
        # Use tqdm for a progress bar, passing the total number of lines
        for item in tqdm(reader, total=num_lines, desc='Processing JSON objects'):
            # Extract the values of the desired keys if they exist
            asin = item.get("asin", "N/A")
            also_buy = ' '.join(item.get("also_buy", []))
            also_view = ' '.join(item.get("also_view", []))

            # Create a list with asin and non-empty fields
            if also_buy or also_view:
                output_values = [asin]
                if also_buy:
                    output_values.append(also_buy)
                if also_view:
                    output_values.append(also_view)
                output_file.write(' '.join(output_values) + '\n')
                
                
# 根据读书最高的64w个点清洗出商品meta
def get_meta_from_highest_degree_nodes(input_file_path, id_list, output_file_path):
    # Count the number of lines for the progress bar (optional but helpful)
    # num_lines = sum(1 for _ in open(input_file_path, 'r'))
    # print("num lines: " + str(num_lines))
        
    # Open the input JSON lines file for reading and the output text file for writing in append mode
    with jsonlines.open(input_file_path) as reader, open(output_file_path, 'w') as output_file:
        # Use tqdm for a progress bar, passing the total number of lines
        for item in tqdm(reader, total=15023059, desc='Processing JSON objects'):
            # Extract the values of the desired keys if they exist
            asin = item.get("asin", "N/A")
            if asin in id_list:
                product_info = {
                    'asin': asin,
                    'title': item.get("title", "N/A"),
                    'description': item.get("description", "N/A"),
                    'brand': item.get("brand", "N/A")
                }
                json_string = json.dumps(product_info, ensure_ascii=False) + "\n"
                output_file.write(json_string)

# with open("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/node_id.txt", 'r') as id_file:
#     id_list = set()
#     for id in id_file:
#         id_list.add(id.strip())
    
# get_meta_from_highest_degree_nodes(input_file_path, id_list, output_file_path)



# 清洗json中的重复id
# input_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta.txt'
# output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta_deduplication.txt'
def clean_repeat_data(input_file_path, output_file_path):
    num_lines = sum(1 for _ in open(input_file_path, 'r'))
    print("num lines: " + str(num_lines))
    id_set = set()
    with jsonlines.open(input_file_path) as reader, open(output_file_path, 'w') as output_file:
        for item in tqdm(reader, total=num_lines, desc='Processing JSON objects'):
            # Extract the values of the desired keys if they exist
            asin = item.get("asin", "N/A")
            if asin not in id_set:
                json_string = json.dumps(item, ensure_ascii=False) + "\n"
                id_set.add(asin)
                output_file.write(json_string)
# clean_repeat_data(input_file_path, output_file_path)



# 清洗64w个节点，只保留meta中出现的30w个
# input_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta_deduplication.txt'
# node_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/node_with_degree_161.txt'
# output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_nodes.txt'
def save_node_in_product_meta(input_file_path, node_file_path, output_file_path):
    num_lines = sum(1 for _ in open(input_file_path, 'r'))
    print("num lines: " + str(num_lines))
    id_set = set()
    
    with jsonlines.open(input_file_path) as reader:
        for item in tqdm(reader, total=num_lines, desc='Processing JSON objects'):
            asin = item.get("asin", "N/A")
            id_set.add(asin)
            
    with open(node_file_path, "r") as node_file , open(output_file_path, 'w') as output_file:
        for line in node_file:
            nodes = line.strip().split()
            # 使用列表推导来过滤不在id_set中的节点
            filtered_nodes = [node for node in nodes if node in id_set]
            # 将过滤后的节点列表转换成空格分隔的字符串
            output_line = ' '.join(filtered_nodes)
            # 写入文件，每行末尾自动包含换行符
            output_file.write(output_line + '\n')
# save_node_in_product_meta(input_file_path, node_file_path, output_file_path)



# 清洗保留后的30w个节点，去除没有度的节点
# input_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_nodes.txt'
# output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/recommendation_network_node.txt'
def clean_missing_data_in_txt(input_file_path, output_file_path):
    with open(input_file_path) as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            nodes = line.strip().split()
            if len(nodes) > 1:
                output_file.write(line)
# clean_missing_data(input_file_path, output_file_path)


# 清洗商品meta，保持商品meta的id和网络id一致
input_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta_deduplication.txt'
node_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/recommendation_network_node.txt'
output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_metadata.txt'
def clean_missing_data_in_json(input_file_path, node_file_path, output_file_path):
    num_lines = sum(1 for _ in open(input_file_path, 'r'))
    print("num lines: " + str(num_lines))
    id_set = set()
    
    with open(node_file_path, "r") as node_file:
        for line in node_file:
            nodes = line.strip().split()
            for node in nodes:
                id_set.add(node)
    
    with jsonlines.open(input_file_path) as reader, open(output_file_path, 'w') as output_file:
        for item in tqdm(reader, total=num_lines, desc='Processing JSON objects'):
            # Extract the values of the desired keys if they exist
            asin = item.get("asin", "N/A")
            if asin in id_set:
                json_string = json.dumps(item, ensure_ascii=False) + "\n"
                output_file.write(json_string)
clean_missing_data_in_json(input_file_path, node_file_path, output_file_path)