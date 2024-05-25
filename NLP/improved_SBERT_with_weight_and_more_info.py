import json
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# 读取数据集
file_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/embedding_data_50_product_meta_more_infos.txt"
data = []
with open(file_path, 'r') as file:
    for line in file:
        product = json.loads(line)
        data.append(product)

# 加载预训练的Sentence-BERT模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def get_sentence_bert_embedding(texts, model):
    embeddings = model.encode(texts, convert_to_tensor=True, device=device)
    return embeddings.cpu().numpy()

def combine_weighted_embedding(products, model,
                               title_weight=2.0, description_weight=1.0, feature_weight=1.5, category_weight=2.0,
                               brand_weight=1.0):
    titles = [product['title'] for product in products]
    descriptions = [' '.join(product['description']) for product in products]
    features = [' '.join(product['feature']) for product in products]
    categories = [' '.join(product['category']) for product in products]
    brands = [product['brand'] for product in products]

    title_embeddings = get_sentence_bert_embedding(titles, model)
    description_embeddings = get_sentence_bert_embedding(descriptions, model)
    feature_embeddings = get_sentence_bert_embedding(features, model)
    category_embeddings = get_sentence_bert_embedding(categories, model)
    brand_embeddings = get_sentence_bert_embedding(brands, model)

    combined_embeddings = (title_weight * title_embeddings +
                           description_weight * description_embeddings +
                           feature_weight * feature_embeddings +
                           category_weight * category_embeddings +
                           brand_weight * brand_embeddings)

    return combined_embeddings

# 为每个产品计算加权嵌入
batch_size = 16  # 定义一个合适的批次大小
embeddings = []
for i in tqdm(range(0, len(data), batch_size), desc="Computing weighted Sentence-BERT embeddings"):
    batch_products = data[i:i+batch_size]
    batch_embeddings = combine_weighted_embedding(batch_products, model)
    embeddings.append(batch_embeddings)

# 将所有嵌入从三维数组转换为二维数组
embeddings = np.vstack(embeddings)

# 保存嵌入数据
embeddings_save_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/improved_SBERT_embeddings_data_50_product_meta_more_infos.npy"
np.save(embeddings_save_path, embeddings)

# 假设新产品信息
new_product = {
    "title": "750 ml Emerald Green Claret/Bordeaux Bottles, 12 per case",
    "description": [
        "Great wine bottle. Standard opening for number 8 or number 9 Cork. Actual color of bottle is darker than photo."],
    "feature": [],
    "category": ["Home & Kitchen", "Kitchen & Dining", "Dining & Entertaining", "Bar Tools & Drinkware"],
    "brand": "Northern Brewer"
}

new_product_embedding = combine_weighted_embedding(
    [new_product],  # 这里传入一个列表
    model
)

# 将新产品的嵌入转换为二维数组
new_product_embedding = new_product_embedding.reshape(1, -1)

# 计算余弦相似度
similarities = cosine_similarity(new_product_embedding, embeddings).flatten()

# 找出最相似的前3个产品及其相似度
top_3_indices = similarities.argsort()[-3:][::-1]
top_3_similarities = similarities[top_3_indices]
top_3_products = [data[idx] for idx in top_3_indices]

print("Top 3 similar products:")
for i, (product, similarity) in enumerate(zip(top_3_products, top_3_similarities)):
    print(f"Rank {i + 1}:")
    print(f"Product: {product}")
    print(f"Similarity: {similarity}")
