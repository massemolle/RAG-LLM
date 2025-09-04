from transformers import BertTokenizerFast, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss

def getEmbed(query) : 
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = 'cpu'
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)

    # Tokeniser le texte
    inputs = tokenizer(query, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    return last_hidden_states.mean(dim=1).cpu().numpy()


def getDocuments(documents, query, k=5, embed = None):
    """
    documents : list[str]
    embed : is embeding needed. This is to avoid converting documents for each query in eval
    """
    if embed is None: 
        document_embeddings = np.vstack([getEmbed(doc) for doc in documents])
    else : 
        document_embeddings = embed
    query_embedding = getEmbed(query)

# Calculer la similarité cosinus entre la requête et les documents
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

# Trier les documents par ordre de similarité décroissante
    most_similar_doc_index = np.argsort(similarities)[::-1]

# Afficher les documents les plus similaires
    """
    for idx in most_similar_doc_index:
        print(f"Similarité : {similarities[idx]:.4f} - Document : {documents[idx]}")
    """
    L = {"doc": [], "score":[]}
    for i, idx in enumerate(most_similar_doc_index):
        L["doc"].append(documents[idx])
        L["score"].append(similarities[idx].item())

        if i==k-1:
            break
    return L
def get_colbert_embeddings(text):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.numpy()

def get_colbert_documents(query, documents, k=5):
    # Obtenir les embeddings pour les documents et la requête
    document_embeddings = np.vstack([get_colbert_embeddings(doc) for doc in documents])
    query_embedding = get_colbert_embeddings(query)

    # Utiliser FAISS pour la recherche approximative des voisins les plus proches
    dimension = query_embedding.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(document_embeddings.reshape(-1, dimension))

    # Rechercher les documents les plus similaires
    k = len(documents)  # Nombre de documents à retourner
    distances, indices = index.search(query_embedding.reshape(-1, dimension), k)

    # Afficher les documents les plus similaires
    """
    for i in range(k):
        doc_index = indices[i] // document_embeddings.shape[1]
        print(f"Similarité : {distances[i]:.4f} - Document : {documents[doc_index]}")"
    """

if __name__ == '__main__':
    documents = [
    "The cat is sitting on the mat.",
    "Dogs are a man's best friend.",
    "I love programming in Python.",
    "The weather is nice today.",
    "I can't believe snake can devour pets",
    "My tiger has an aura dept, he lacks the skibidy rizz",
    "Donc là le serpent a graille mon félin ??"
    ]
    query = 'Un python a mangé mon chat'

    print(getDocuments(documents, query, k=10))
    