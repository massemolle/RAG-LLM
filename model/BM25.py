import bm25s
import os
import database as db

class BM25():
    def __init__(self, k=5):

        self.k = k
        self.retriever = bm25s.BM25()


    def index(self,corpus):
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        self.retriever.index(corpus_tokens)

    def answer(self, query):
        query_tokens = bm25s.tokenize(query)

        # Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
        # To return docs instead of IDs, set the `corpus=corpus` parameter.
        results, scores = self.retriever.retrieve(query_tokens, k=self.k)
        L={'doc': [], 'score': []}
        for i in range(results.shape[1]):
            doc, score = results[0, i], scores[0, i]
            #print(f"Rank {i+1} (score: {score:.2f}): {doc}")
            L['doc'].append(doc)
            L['score'].append(score)
        return L


    def save(self, path, corpus):
        self.retriever.save(os.path.join(path,"index_bm25"), corpus=corpus)


    def load(self,path):
        self.retriever = bm25s.BM25.load(os.path.join(path,"index_bm25"), load_corpus=True)
         

if __name__ == '__main__':
    path = '/home/epy/code/database/docx/test'
    if 1 :
        model = BM25()
        
        docs = db.doc2Text(path)
        corpus = []
        for d in docs : 
            corpus = corpus + d[0]
        print('corpus')
        model.index(corpus)
        print('done')
        
    print(model.answer('chain of'))