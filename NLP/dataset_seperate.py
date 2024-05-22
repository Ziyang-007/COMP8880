import json
import random

# 读取数据集文件
with open('C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/50_product_meta_more_infos.txt', 'r') as f:
    lines = f.readlines()

# 随机选择1000条数据
random.shuffle(lines)
selected_lines = lines[:28806]
remaining_lines = lines[28806:]

# 将选择的1000条数据保存到一个新的文件中
with open('C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/test_data_50_product_meta_more_infos.txt', 'w') as f:
    for line in selected_lines:
        f.write(line)

# 将剩余的数据保存到另一个文件中
with open('C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/embedding_data_50_product_meta_more_infos.txt', 'w') as f:
    for line in remaining_lines:
        f.write(line)
