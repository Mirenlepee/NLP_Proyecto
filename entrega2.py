# SOLO INSTALAR UNA VEZ EN LA TERMINAL
# pip install nltk
# pip install pandas numpy gensim scikit-learn torch transformers matplotlib wordcloud tqdm

# --- 1. LIBRERÍAS NECESARIAS ---
import os
import pandas as pd
import numpy as np
import re
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, FastText
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import torch

# --- 2. DESCARGA DE RECURSOS NLTK (solo la primera vez) ---
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

# --- 3. CARGA DE DATOS ---
# Cambia esta ruta si tu estructura de carpetas es distinta
base_path = "ted_transcripts/ted_transcripts"

texts = []
for file in os.listdir(base_path):
    if file.endswith(".txt"):
        with open(os.path.join(base_path, file), "r", encoding="utf-8") as f:
            texts.append(f.read())

print(f"Total de transcripciones cargadas: {len(texts)}")

# Crear un DataFrame
df = pd.DataFrame({"text": texts})
print(df.head())


# --- 4. LIMPIEZA Y PREPROCESAMIENTO DEL TEXTO ---

def clean_text(text):
    text = text.lower()                          # minúsculas
    text = re.sub(r"\[.*?\]", "", text)          # eliminar contenido entre corchetes
    text = re.sub(r"http\S+|www\S+", "", text)   # eliminar URLs
    text = re.sub(r"[^a-z\s]", "", text)         # eliminar caracteres no alfabéticos
    text = re.sub(r"\s+", " ", text).strip()     # eliminar espacios extras
    return text

df["clean_text"] = df["text"].apply(clean_text)

# Tokenización
df["tokens"] = df["clean_text"].apply(nltk.word_tokenize)

# Eliminar stopwords y lematizar
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_tokens(tokens):
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

df["tokens"] = df["tokens"].apply(preprocess_tokens)
print("✅ Textos preprocesados correctamente.")

# --- 5. ANÁLISIS EXPLORATORIO ---
all_words = [word for tokens in df["tokens"] for word in tokens]
freq_dist = nltk.FreqDist(all_words)
freq_df = pd.DataFrame(freq_dist.most_common(20), columns=["Word", "Frequency"])


# Carpeta donde se guardarán las gráficas
output_folder = "resultados_graficas"
# os.makedirs(output_folder, exist_ok=True)  # crea la carpeta si no existe

# Top palabras
plt.figure(figsize=(10,5))
plt.bar(freq_df["Word"], freq_df["Frequency"])
plt.title("Palabras más frecuentes en TED Talks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "top_palabras.png"))  # guarda la imagen
plt.show()

# Nube de palabras
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "nube_palabras.png"))  # guarda la imagen
plt.show()


# --- 6. REPRESENTACIÓN TRADICIONAL: Bag of Words ---
vectorizer_bow = CountVectorizer(max_features=5000)
X_bow = vectorizer_bow.fit_transform(df["clean_text"])
print(f"BoW matrix shape: {X_bow.shape}")

# --- 7. REPRESENTACIÓN TRADICIONAL: TF-IDF ---
vectorizer_tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer_tfidf.fit_transform(df["clean_text"])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")


# --- 8. EMBEDDINGS NO CONTEXTUALES (Word2Vec / FastText) ---

print("Entrenando modelo Word2Vec...")
w2v_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=3, workers=4)
w2v_model.save("word2vec_ted.model")

print("Entrenando modelo FastText...")
fasttext_model = FastText(sentences=df["tokens"], vector_size=100, window=5, min_count=3, workers=4)
fasttext_model.save("fasttext_ted.model")

# Ejemplo de resultados
try:
    print(w2v_model.wv.most_similar("technology", topn=5))
except KeyError:
    print("Palabra 'technology' no encontrada en el vocabulario Word2Vec.")

# # --- 9. EMBEDDINGS CONTEXTUALES (BERT) ---
# print("Generando embeddings BERT (esto puede tardar unos segundos)...")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# def get_bert_embedding(text):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# # Ejemplo: embedding de la primera transcripción
# bert_vector = get_bert_embedding(df["clean_text"][0])
# print("Dimensión del embedding BERT:", bert_vector.shape)

# # --- 10. ANÁLISIS DE PALABRAS NO REPRESENTADAS ---
# vocab_words = set(w2v_model.wv.index_to_key)
# oov_words = [w for w in all_words if w not in vocab_words]
# oov_ratio = len(oov_words) / len(all_words)
# print(f"Palabras fuera de vocabulario (Word2Vec): {oov_ratio:.2%}")

# # --- 11. GUARDAR DATOS PREPROCESADOS ---
# df.to_csv("tedtalks_preprocessed.csv", index=False, encoding="utf-8")
# print("✅ Archivo guardado: tedtalks_preprocessed.csv")