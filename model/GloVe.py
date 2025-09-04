import numpy as np
from collections import defaultdict
import re

class GloVe:
    def __init__(self, vector_size, window_size=5, learning_rate=0.05, epochs=50):
        self.vector_size = vector_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize word vectors and biases
        self.word_vectors = None
        self.word_biases = None
        self.context_biases = None

        # Co-occurrence matrix
        self.cooccurrence_matrix = None

    def fit(self, corpus):
        # Map corpus to indices
        self.word_to_index, self.index_to_word, mapped_corpus = self.map_corpus_to_indices(corpus)
        self.vocab_size = len(self.word_to_index)

        # Initialize word vectors and biases
        self.word_vectors = np.random.rand(self.vocab_size, self.vector_size)
        self.word_biases = np.zeros(self.vocab_size)
        self.context_biases = np.zeros(self.vocab_size)

        # Co-occurrence matrix
        self.cooccurrence_matrix = np.zeros((self.vocab_size, self.vocab_size))

        # Build co-occurrence matrix
        self._build_cooccurrence_matrix(mapped_corpus)

        # Train GloVe model
        self._train()

    def map_corpus_to_indices(self, corpus):
        word_to_index = {}
        index_to_word = []
        mapped_corpus = []

        # Build vocabulary
        word_count = defaultdict(int)
        for sentence in corpus:
            for word in sentence:
                word_count[word] += 1

        # Map words to indices
        for word in word_count:
            word_to_index[word] = len(word_to_index)
            index_to_word.append(word)

        # Map corpus to indices
        for sentence in corpus:
            mapped_sentence = [word_to_index[word] for word in sentence]
            mapped_corpus.append(mapped_sentence)

        return word_to_index, index_to_word, mapped_corpus

    def _build_cooccurrence_matrix(self, corpus):
        for sentence in corpus:
            for i, center_word in enumerate(sentence):
                context_words = sentence[max(0, i - self.window_size): i] + sentence[i + 1: min(len(sentence), i + self.window_size + 1)]
                for context_word in context_words:
                    if center_word == context_word:
                        continue
                    if not (isinstance(center_word, int) and isinstance(context_word, int)):
                        raise ValueError(f"Indices must be integers. center_word: {center_word}, context_word: {context_word}")
                    if center_word >= self.vocab_size or context_word >= self.vocab_size:
                        raise ValueError(f"Indices out of range. center_word: {center_word}, context_word: {context_word}, vocab_size: {self.vocab_size}")
                    self.cooccurrence_matrix[center_word, context_word] += 1 / abs(i - sentence.index(context_word))

    def _train(self):
        for epoch in range(self.epochs):
            for i in range(self.vocab_size):
                for j in range(self.vocab_size):
                    if self.cooccurrence_matrix[i, j] > 0:
                        weight = min(1.0, (self.cooccurrence_matrix[i, j] / 100) ** 0.75)
                        cost_inner = np.dot(self.word_vectors[i], self.word_vectors[j]) + self.word_biases[i] + self.context_biases[j] - np.log(self.cooccurrence_matrix[i, j])
                        grad_squared = 2 * weight * cost_inner

                        # Update word vectors and biases
                        self.word_vectors[i] -= self.learning_rate * grad_squared * self.word_vectors[j]
                        self.word_vectors[j] -= self.learning_rate * grad_squared * self.word_vectors[i]
                        self.word_biases[i] -= self.learning_rate * grad_squared
                        self.context_biases[j] -= self.learning_rate * grad_squared

    def get_word_vector(self, word):
        if word in self.word_to_index:
            word_index = self.word_to_index[word]
            return self.word_vectors[word_index]
        else:
            return None

def get_corpus(text):
    V = ''
    C = []
    for phrase in text : 
        phrase = phrase.lower()
        phrase = re.sub(r'[^\w\s]','',phrase)
        C.append(phrase.split(" "))
        V = V + " " +phrase
    return V[1:], C    # Remove the initial space 

if __name__ == "__main__":

    documents = [
    "The cat is sitting on the mat.",
    "Dogs are a man's best friend.",
    "I love programming in Python.",
    "The weather is nice today.",
    "I can't believe snake can devour pets",
    "My tiger has an aura dept, he lacks the skibidy rizz",
    "Donc là le serpent a foudroyé mon félin??"
    ]

    vocab, corpus = get_corpus(documents)
    vocab = vocab.split(" ")

    vocab_size = len(set(vocab))
    vector_size = 8 

    glove = GloVe(vocab_size, vector_size)
    glove.fit(corpus)

    word = "cat"
    word_vector = glove.get_word_vector(word)
    print(f"Word vector for '{word}': {word_vector}")