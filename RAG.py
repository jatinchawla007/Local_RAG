import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from sentence_transformers import SentenceTransformer # Assuming you used this for embeddings

# --- Configuration ---
EMBEDDINGS_CSV_PATH = 'text_chunks_and_embeddings_df.csv' # Path to your CSV file
LM_STUDIO_API_URL = "http://192.168.0.187:1234/v1/chat/completions" # Default LM Studio API endpoint
EMBEDDING_MODEL_NAME = 'all-mpnet-base-v2' # The name of the embedding model you used for your chunks

# Load the embedding model (make sure this matches the one you used for your document chunks)
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    print(f"Error loading embedding model: {e}. Make sure you have the correct model name and internet connection.")
    print("If you used a different embedding method (e.g., OpenAI), you'll need to adapt this part.")
    exit()

# --- 1. Load your Embeddings ---
def load_embeddings_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Convert string representation of list/array to actual numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    return df

# --- 2. Embed User Query ---
def embed_query(query):
    return embedding_model.encode(query)

# --- 3. Vector Search (Similarity Search) ---
def find_most_similar_chunks(query_embedding, df_embeddings, top_k=3):
    embeddings_matrix = np.vstack(df_embeddings['embedding'].values)
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings_matrix)
    
    # Get the indices of the top_k most similar chunks
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    
    # Retrieve the corresponding chunks and their texts
    retrieved_chunks = df_embeddings.iloc[top_k_indices]
    
    # Return a list of chunk texts
    return retrieved_chunks['sentence_chunk'].tolist()

# --- 4. LM Studio Setup & API Call ---
def generate_response_with_gemma(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, headers=headers, json=data)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to LM Studio API at {LM_STUDIO_API_URL}.")
        print("Please ensure LM Studio is running and the Gemma 3 model is loaded.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling LM Studio API: {e}")
        return None

# --- Main RAG Loop ---
if __name__ == "__main__":
    print("Loading embeddings...")
    document_embeddings_df = load_embeddings_from_csv(EMBEDDINGS_CSV_PATH)
    print(f"Loaded {len(document_embeddings_df)} document chunks.")

    # Start LM Studio and load the Gemma 3 model before running this script
    print("\n--- LM Studio RAG Demo with Gemma 3 ---")
    print("Ensure LM Studio is running and serving Gemma 3 on port 1234.")

    while True:
        user_query = input("\nEnter your query (type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        print("Embedding query...")
        query_embedding = embed_query(user_query)

        print("Finding relevant document chunks...")
        relevant_texts = find_most_similar_chunks(query_embedding, document_embeddings_df, top_k=4) # Adjust top_k as needed

        if not relevant_texts:
            print("No relevant chunks found. Please try a different query or check your embeddings.")
            continue
            
        context = "\n\n".join(relevant_texts)
        print("\n--- Retrieved Context ---")
        for i, text in enumerate(relevant_texts):
            print(f"Chunk {i+1}:\n{text}\n---")
        print("------------------------")

        # --- 5. Prompt Engineering ---
        prompt = f"""Given the following context, please answer the question.
If the answer cannot be found in the context, politely state that you don't have enough information.

Context:
{context}

Question: {user_query}

Answer:"""

        print("\nSending request to Gemma 3 via LM Studio...")
        response = generate_response_with_gemma(prompt)

        if response:
            print("\n--- Gemma 3's Answer ---")
            print(response)
            print("------------------------")