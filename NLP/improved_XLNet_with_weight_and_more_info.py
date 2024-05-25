import json
import torch
from transformers import XLNetTokenizer, XLNetModel
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

# 加载预训练的XLNet模型和tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_xlnet_embedding(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors='pt', max_length=512, truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, -1, :].cpu().numpy()  # 将输出移回CPU并分离计算图


def combine_weighted_embedding(products, model, tokenizer, device,
                               title_weight=2.0, description_weight=1.0, feature_weight=1.5, category_weight=2.0,
                               brand_weight=1.0):
    titles = [product['title'] for product in products]
    descriptions = [' '.join(product['description']) for product in products]
    features = [' '.join(product['feature']) for product in products]
    categories = [' '.join(product['category']) for product in products]
    brands = [product['brand'] for product in products]

    title_embeddings = get_xlnet_embedding(titles, model, tokenizer, device)
    description_embeddings = get_xlnet_embedding(descriptions, model, tokenizer, device)
    feature_embeddings = get_xlnet_embedding(features, model, tokenizer, device)
    category_embeddings = get_xlnet_embedding(categories, model, tokenizer, device)
    brand_embeddings = get_xlnet_embedding(brands, model, tokenizer, device)

    combined_embeddings = (title_weight * title_embeddings +
                           description_weight * description_embeddings +
                           feature_weight * feature_embeddings +
                           category_weight * category_embeddings +
                           brand_weight * brand_embeddings)

    return combined_embeddings


# 为每个产品计算加权嵌入
batch_size = 16  # 定义一个合适的批次大小
embeddings = []
for i in tqdm(range(0, len(data), batch_size), desc="Computing weighted XLNet embeddings"):
    batch_products = data[i:i+batch_size]
    batch_embeddings = combine_weighted_embedding(batch_products, model, tokenizer, device)
    embeddings.append(batch_embeddings)

# 将所有嵌入从三维数组转换为二维数组
embeddings = np.vstack(embeddings)

# 保存嵌入数据
embeddings_save_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/improved_XLNet_embeddings_data_50_product_meta_more_infos.npy"
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
    model,
    tokenizer,
    device
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
