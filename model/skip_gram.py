import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

# LOCAL
import model.database as db

def createVocabulary(corpus : list) -> dict :
    vocabulary = {}
    i = 0
    for s in corpus :
        for w in s.split() :
            if w not in vocabulary : 
                vocabulary[w] = i
                i+=1
    return vocabulary


def createSet(corpus : list, n_gram:int = 1 ) -> list:
    Data = []
    for s in corpus : 
        for i, w in enumerate(s.split()):
            context = []
            for n in range(1, n_gram+1):

                try :
                    context.append(s.split())[i-n]
                    context.append(s.split()[i+n])
                except : 
                    continue
            
            Data.append([w, context])
    return Data


class SkipGram(nn.Module) :
    EMBED_DIM = 1000
    def __init__(self, vocab_size : int) : 
        super(SkipGram, self).__init__()
        self.linear = nn.Linear(in_features = self.EMBED_DIM, out_features = vocab_size)
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.EMBED_DIM)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(F.relu(x))
        return x
    

def train(corpus : list) : 

    vocab = createVocabulary(corpus)
    print(f'Len of vocabulary : {len(vocab)}')
    data = createSet(corpus)

    model = SkipGram(len(vocab))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    lossF = nn.CrossEntropyLoss()

    Epoch = 20
    L = []
    for epoch in range(Epoch) : 
        total_loss =0
        for word, context in data :
            try : 
                inp = torch.tensor(vocab[word], dtype = torch.long)
                try :   #
                    label = torch.tensor([vocab[w] for w in context], dtype = torch.long)
                except TypeError:   # Case where context is a list with a single element being the actual context
                    label = torch.tensor([vocab[w[0]] for w in context], dtype = torch.long)
            except KeyError : 
                raise KeyError("Trying to id unknow word in vocabulary")
            optimizer.zero_grad()
            output = model(inp)
            loss = lossF(output.view(1,-1), label)
            loss.backward()
            optimizer.step()
        

        total_loss += loss.item()
        L.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

    torch.save(model.state_dict(), "./model_w/skipgram/skipgram.pth")
    T = [i for i in range(len(L))]

    plt.plot(T, L)
    plt.show()




if __name__=='__main__' : 
    
    doc = db.cleanDocx('./bin/test.docx')
    corpus = db.docToCorpus(doc)
    
    print("corpus done")
    train(corpus)
    
  