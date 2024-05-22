import json
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# 加载嵌入数据集
file_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/embedding_data_50_product_meta_more_infos.txt"
embeddingData = []
with open(file_path, 'r') as file:
    for line in file:
        product = json.loads(line)
        embeddingData.append(product)

# 加载预训练的BERT模型和tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载嵌入数据，并移动到GPU
embeddings_load_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/embeddings_data_50_product_meta_more_infos.npy"
embeddings = np.load(embeddings_load_path)
embeddings = torch.tensor(embeddings, device=device)

def get_distilbert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到GPU
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # 输出在GPU上

def combine_weighted_embedding(title, description, feature, category, brand, model, tokenizer, device,
                               title_weight=2.0, description_weight=1.0, feature_weight=1.5, category_weight=2.0,
                               brand_weight=1.0):
    title_embedding = get_distilbert_embedding(title, model, tokenizer, device)
    description_embedding = get_distilbert_embedding(description, model, tokenizer, device)
    feature_embedding = get_distilbert_embedding(feature, model, tokenizer, device)
    category_embedding = get_distilbert_embedding(category, model, tokenizer, device)
    brand_embedding = get_distilbert_embedding(brand, model, tokenizer, device)

    combined_embedding = (title_weight * title_embedding +
                          description_weight * description_embedding +
                          feature_weight * feature_embedding +
                          category_weight * category_embedding +
                          brand_weight * brand_embedding)

    return combined_embedding

# 根据ASIN查找产品
def find_product_by_asin(data, asin):
    for product in data:
        if product['asin'] == asin:
            return product
    return None

# 给定ASIN，计算相似度并返回最相似的前3个产品
def find_top_3_similar_products(asin, testData, embeddingData, embeddings, model, tokenizer, device):
    product = find_product_by_asin(testData, asin)
    if not product:
        return None

    title = product['title']
    description = ' '.join(product['description'])
    feature = ' '.join(product['feature'])
    category = ' '.join(product['category'])
    brand = product['brand']

    product_embedding = combine_weighted_embedding(title, description, feature, category, brand, model, tokenizer, device)
    product_embedding = product_embedding.unsqueeze(0)  # 增加一个batch维度

    # 计算余弦相似度
    similarities = torch.nn.functional.cosine_similarity(product_embedding, embeddings.unsqueeze(0), dim=-1).squeeze(0)
    top_3_indices = similarities.argsort(descending=True)[:3]
    top_3_similar_products = [embeddingData[idx] for idx in top_3_indices]

    return top_3_similar_products

# 从文件加载测试集
test_file_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/test_data_50_product_meta_more_infos.txt"
test_set = []
with open(test_file_path, 'r') as test_file:
    for line in test_file:
        test_product = json.loads(line)
        test_set.append(test_product)

# 查找每个测试集商品的前3个相似商品，并保存结果到文件
output_file_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/similar_products_data_50_product_meta_more_infos.txt"
with open(output_file_path, 'w') as output_file:
    for test_product in tqdm(test_set, desc="Processing test set"):
        test_asin = test_product['asin']
        top_3_products = find_top_3_similar_products(test_asin, test_set, embeddingData, embeddings, model, tokenizer, device)

        if top_3_products:
            similar_asins = [test_asin] + [product['asin'] for product in top_3_products]
            output_file.write(' '.join(similar_asins) + '\n')
        else:
            output_file.write(test_asin + ' Not Found\n')
