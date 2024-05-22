import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load the dataset
data_path = '/Users/fengziyang/Desktop/ANU/COMP8880-NetworkScience/Project/COMP8880/dataset/50_product_meta_more_infos.txt'
df = pd.read_json(data_path, lines=True)

# Select the relevant column
category_text = df['category']

# Handle NaN values and ensure all entries are strings
category_text = category_text.fillna('').astype(str)

# Flatten the list of categories if they are nested lists
category_text = category_text.explode()

# Tokenization and stopwords removal
# nltk.download('punkt')
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_tokenize(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords
    return filtered_words

# Applying the tokenization and cleaning function
all_words = category_text.apply(clean_tokenize).explode()

# Counting word frequencies
word_freq = Counter(all_words)

# Getting the 20 most common words
most_common_words = word_freq.most_common(50)

# Display the most common words with their frequencies
for word, frequency in most_common_words:
    print(f"{word}: {frequency}")
