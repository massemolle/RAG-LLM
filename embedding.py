import bm25s
import os
from transformers import BertTokenizerFast, BertModel
import torch
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity


# Embeding method template, no need to super
class Method():
    def __init__(self, k=5):
        self.k = k

    def retrieve(self, query : str , path = None, doc = None) -> dict:
        '''
        Input : 
        query : str -> Answer to give answer to
        path : str -> Where documents or embeddings are stored locally
        doc : ... -> list of documents or document to extract context from

        Return : 
        dict with 2 keys : 
            - 'doc' with a list of str, corresponding to the context
            - 'score' the similarity scores, used to set a threshold of minimum similarity to accept the context
        ''' 
        return ['']
    
    def process(self, doc = None, path = None, get = False) : 
        '''
        Input : 
        path : str -> Where to load the documents from. 
        doc : ... -> list of documents or document to pre-process
        Return : 
        None if get==False else (documents embeddings, docs)
        ''' 
        return None
    

class BM25():
    def __init__(self, k=5, path=None):
        self.k = k
        
        if path is not None and 'index_bm25' in os.listdir(path):
            self.retriever = bm25s.BM25.load(os.path.join(path,"index_bm25"), load_corpus=True)
        else : 
            self.retriever = bm25s.BM25()

    def retrieve(self, query, doc=None, path=None):
        query_tokens = bm25s.tokenize(query)
        
        if self.retriever.corpus is None : # Making sure the corpus is correctly loaded, and in the right format
            self.retriever = bm25s.BM25.load(os.path.join(path,"index_bm25"), load_corpus=True)
        if len(self.retriever.corpus) == 1:
            self.retriever.corpus = self.retriever.corpus[0]
            if len(self.retriever.corpus) == 2:
                self.retriever.corpus = self.retriever.corpus[0]
        
        try:
            results, scores = self.retriever.retrieve(query_tokens, k=self.k, corpus=self.retriever.corpus)
        except Exception as e:
            print(f'Error getting scores back: {e}')
            return {'doc': [], 'score': []}
        
        L = {'doc': [], 'score': []}
        for i in range(len(results[0])):
            doc, score = results[0][i], scores[0][i]
            L['doc'].append(doc)
            L['score'].append(score)
        return L
    
    def process(self, doc = None, path = None, get = False):
        if len(doc[0]) == 2:   # Doc is [file_content, file_name]
            new_doc=[]
            for d in doc : 
                new_doc = new_doc +d[0]
        else : 
            new_doc = doc
        corpus_tokens = bm25s.tokenize(new_doc, stopwords="en")

        self.retriever.index(corpus_tokens)
        if path is not None :
            self.retriever.save(os.path.join(path,"index_bm25"), corpus=doc)


class BERT():
    def __init__(self, k=5, device = 'cpu', path=None):
        self.k = k
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.device=device
        self.model = self.model.to(self.device)

        if path is not None and "temp_emb.pkl" in os.listdir(path):
            with open(os.path.join(path, "temp_emb.pkl"), "rb") as fIn:
                    cache_data = pickle.load(fIn)
                    self.embed = cache_data['embeddings']

    def retrieve(self, query : str , path = None, doc = None) -> list[str]:
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        if self.device == 'cpu':
            query_embedding = last_hidden_states.mean(dim=1).cpu().numpy()
        else :
            query_embedding = last_hidden_states.mean(dim=1)

        similarities = cosine_similarity(query_embedding, self.embed).flatten()

        # Trier les documents par ordre de similarité décroissante
        most_similar_doc_index = np.argsort(similarities)[::-1]

        # Afficher les documents les plus similaires

        L = {"doc": [], "score":[]}
        for i, idx in enumerate(most_similar_doc_index):
            L["doc"].append(doc[idx])
            L["score"].append(similarities[idx].item())

            if i==self.k-1:
                break
        return L
    
    def process(self, doc = None, path = None, get = False) : 
        e_list = []
        for d in doc : 

            # Tokeniser le texte
            inputs = self.tokenizer(d, return_tensors='pt',padding=True, truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            last_hidden_states = outputs.last_hidden_state

            e_list.append(last_hidden_states.mean(dim=1).cpu().numpy())
        document_embeddings = np.vstack(e_list)
        if path is not None :
            with open(os.path.join(path, "temp_emb.pkl"), "wb") as fOut:    #Save embedding
                pickle.dump({'embeddings': document_embeddings, 'wiki_page' : doc}, fOut)