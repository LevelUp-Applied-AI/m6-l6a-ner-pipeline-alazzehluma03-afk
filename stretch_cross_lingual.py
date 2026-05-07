import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

#load data
df = pd.read_csv("data/climate_articles.csv")

print(df.head())
print(df["language"].value_counts())

#filter for English and Arabic language articles
english_df = df[df["language"] == "en"].copy()
arabic_df = df[df["language"] == "ar"].copy()

print("English articles:", len(english_df))
print("Arabic articles:", len(arabic_df))

#Select 10 articles from each language for analysis
english_texts = english_df["text"].dropna().tolist()[:10]
arabic_texts = arabic_df["text"].dropna().tolist()[:10]

#Load multilingual model BERT
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()

#Mean pooling function to get sentence embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state

    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1),
        min=1e-9
    )

#Embedding  extracting function
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"])

    return embedding.squeeze().numpy()

#generate embeddings for English and Arabic texts
english_embeddings = [get_embedding(text) for text in english_texts]
arabic_embeddings = [get_embedding(text) for text in arabic_texts]

#Combine embeddings 
all_embeddings = english_embeddings + arabic_embeddings
similarity_matrix = cosine_similarity(all_embeddings)

#Create labels for the heatmap
english_labels = [
    "EN: " + text[:40].replace("\n", " ")
    for text in english_texts
]

arabic_labels = [
    "AR: " + text[:40].replace("\n", " ")
    for text in arabic_texts
]

labels = english_labels + arabic_labels

#Heatmap visualization
plt.figure(figsize=(14, 12))

sns.heatmap(
    similarity_matrix,
    xticklabels=labels,
    yticklabels=labels,
    cmap="coolwarm"
)

plt.title("Cross-Lingual Similarity Heatmap")
plt.xticks(rotation=90)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("cross_lingual_similarity_heatmap.png")
plt.close()

#Example Similarity Comparison
cross_similarity = cosine_similarity(
    [english_embeddings[0]],
    [arabic_embeddings[0]]
)[0][0]

within_similarity = cosine_similarity(
    [english_embeddings[0]],
    [english_embeddings[1]]
)[0][0]

print("Cross-lingual similarity:", cross_similarity)
print("Within-language similarity:", within_similarity)