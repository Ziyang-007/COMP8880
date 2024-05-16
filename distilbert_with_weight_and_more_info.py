import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

# 读取数据集
file_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/dataset/product_meta_more_infos.txt"
data = []
with open(file_path, 'r') as file:
    for line in file:
        product = json.loads(line)
        data.append(product)

# 加载预训练的BERT模型和tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 将模型移动到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_distilbert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs = {key: value.to(device) for key, value in inputs.items()}  # 将输入移动到GPU
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().detach().numpy()  # 将输出移回CPU并分离计算图


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


# 为每个产品计算加权嵌入
embeddings = []
for product in tqdm(data, desc="Computing weighted DistilBERT embeddings"):
    title = product['title']
    description = ' '.join(product['description'])
    feature = ' '.join(product['feature'])
    category = ' '.join(product['category'])
    brand = product['brand']
    embedding = combine_weighted_embedding(title, description, feature, category, brand, model, tokenizer, device)
    embeddings.append(embedding)

# 将所有嵌入从三维数组转换为二维数组
embeddings = np.vstack(embeddings)

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
    new_product['title'],
    ' '.join(new_product['description']),
    ' '.join(new_product['feature']),
    ' '.join(new_product['category']),
    new_product['brand'],
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

# 保存训练完的模型
model_save_path = "C:/Users/Zeyi/PycharmProjects/COMP8880/distilbert_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
