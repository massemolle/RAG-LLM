import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
import json
import torch


from model.database import docToCorpus, doc2Text
# Sample corpus
    

# Generate training data
def generate_training_data(corpusinit,path, window_size=2):
    if path is not None : 
        with open(path, 'r') as f :
            doc = f.read()
            corpus = docToCorpus(doc)
    if corpusinit is not None : 
        corpus = corpusinit
    L=[]
    for phrase in corpus :
        L = L + phrase.split(" ")

    corpus = L

    # Build vocabulary
    vocab = set(corpus)
    vocab_size = len(vocab)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    data = []
    for i in range(window_size, len(corpus) - window_size):
        context = [corpus[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = corpus[i]
        data.append((context, target))
    return data, word_to_idx, idx_to_word, vocab_size


# Convert words to indices
def word_to_tensor(word, word_to_idx):
    return torch.tensor(word_to_idx[word], dtype=torch.long)

def context_to_tensor(context, word_to_idx):
    return torch.tensor([word_to_idx[word] for word in context], dtype=torch.long)

# CBOW Model
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, context):
        embedded = self.embeddings(context)
        context_vector = embedded.mean(dim=0)
        output = self.linear(context_vector)
        return output

def train(path = None, corpus = None):
    # Get data
    
    data, word_to_idx, idx_to_word, vocab_size = generate_training_data(corpus, path)
    # Hyperparameters
    embedding_dim = 10
    model = CBOW(vocab_size, embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training
    num_epochs = 200
    for epoch in range(num_epochs):
        total_loss = 0
        for context, target in data:
            context_tensor = context_to_tensor(context, word_to_idx)
            target_tensor = word_to_tensor(target, word_to_idx).unsqueeze(0)
        
            optimizer.zero_grad()
            output = model(context_tensor)
            loss = model.loss_fn(output.view(1, -1), target_tensor)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item()
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}')

    # Code exemple to access word embeddings ! 
    """
    test_e = ['crazy']
    test_t = context_to_tensor(test_e, word_to_idx)
    print(model.embeddings(test_t))
    """

    torch.save(model.state_dict(), "./model_w/cbow/cbow.pth")
    param = {'w2i' : word_to_idx, 'i2w' : idx_to_word, 'vSize' : vocab_size}
    with open("./model_w/cbow/cbow.json", 'w') as f:
        json.dump(param, f)
    print("Saved !")
    return word_to_idx, idx_to_word, vocab_size, model


def predict(context_words, idx_to_word,word_to_idx, model):
    # Loading model

    context_tensor = context_to_tensor(context_words, word_to_idx)
    with torch.no_grad():
        output = model(context_tensor)
        predicted_idx = torch.argmax(output).item()
    return idx_to_word[str(predicted_idx)]



if __name__ == '__main__':
    L = doc2Text("./database/docx/train")
    T= []
    for t, doc in L:
        T = T + t
    
    word_to_idx, idx_to_word, vocab_size, model= train(corpus = T)
    print("Training completed!")
    """
    embedding_dim = 10
    model = CBOW(vocab_size, embedding_dim)
    model.load_state_dict(torch.load("cbow_model.pth"))
    model.eval()
    w= predict(['i', 'was', 'crazy'], idx_to_word, word_to_idx, model)
    print(w)
    """
