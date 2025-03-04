import numpy as np
import pandas as pd
import os
import faiss
from openai import OpenAI

# Setează clientul OpenAI
client = OpenAI()

# Funcție pentru a obține embedding pentru un text
def get_embedding(text, model_name="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        emb = client.embeddings.create(input=[text], model=model_name)
        return emb.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for text: {text}")
        print(e)
        return None  # Sau returnați un vector nul

# Verifică dacă embeddings-urile și indexul sunt deja salvate
if os.path.exists('embeddings.npy') and os.path.exists('index.faiss'):
    # Încarcă embeddings din fișierul salvat
    embeddings = np.load('embeddings.npy')

    # Încarcă indexul FAISS salvat
    index = faiss.read_index('index.faiss')

else:
    # Dacă embeddings-urile nu sunt salvate, procedăm cu procesul normal (calcularea și salvarea embeddings)
    df = pd.read_csv("bbc_text_cls.csv")
    df_small = df.sample(100)  # Folosim doar 100 de documente pentru exemplu

    # Creează embeddings pentru documente (apel API OpenAI)
    df_small['embeddings'] = df_small['text'].apply(get_embedding)
    embeddings = np.array(df_small['embeddings'].tolist())

    # Salvează embeddings într-un fișier pentru reutilizare
    np.save('embeddings.npy', embeddings)

    # Crează indexul FAISS
    dims = len(embeddings[0])  # Calculăm dimensiunea vectorilor
    index = faiss.IndexFlatL2(dims)
    index.add(embeddings)

    # Salvează indexul FAISS într-un fișier pentru reutilizare
    faiss.write_index(index, 'index.faiss')

# Funcție pentru a căuta documente relevante folosind FAISS
def search_document(query, k=5):
    query_emb = get_embedding(query)
    query_emb = np.array(query_emb).reshape(1, -1)
    distances, indices = index.search(query_emb, k)
    return indices[0]

# Funcție pentru a crea completări folosind OpenAI GPT-3.5
def complete(user_prompt, max_tokens=100):
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return completion

# Funcție pentru a genera un răspuns bazat pe întrebare și context
def qa(question, context):
    prompt = f"""Please answer the question given the provided context.
    Question:
    ```
    {question}
    ```
    Context:
    ```
    {context}
    ```
    """
    completion = complete(prompt)
    content = completion.choices[0].message.content
    print(content)

########## DEMO: Întrebarea utilizatorului ################
query = "By what percentage did China economy expand in 2004?"

# Găsește cele mai relevante documente
indices = search_document(query)

# Arată răspunsul generat de GPT-3.5 pe baza documentului relevant
qa(query, df_small.iloc[indices[0]].text)
