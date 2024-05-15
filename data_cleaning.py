import jsonlines
import json
from tqdm import tqdm

# Replace this with the actual file path
input_file_path = '/Volumes/970EVOPLUS/AmazonReviewDataset/All_Amazon_Meta.json'

# Path to your output text file
output_file_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/highest_degree_product_meta.txt'

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
                    'asin': item.get("asin", "N/A"),
                    'title': item.get("title", "N/A"),
                    'feature': item.get("feature", "N/A"),
                    'description': item.get("description", "N/A"),
                    'brand': item.get("brand", "N/A"),
                    'category': item.get("category", "N/A")
                }
                json_string = json.dumps(product_info, ensure_ascii=False) + "\n"
                output_file.write(json_string)



with open("/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/node_id.txt", 'r') as id_file:
    id_list = []
    for id in id_file:
        id_list.append(id.strip())
    
get_meta_from_highest_degree_nodes(input_file_path, id_list, output_file_path)
    