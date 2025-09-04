import torch
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoder, DPRContextEncoderTokenizer
import faiss
import numpy as np


def getDPR(question, corpus, k = 5):
    question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    passage_embeddings = []
    for phrase in corpus:
        inputs = context_tokenizer(phrase, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = context_encoder(**inputs)
        passage_embeddings.append(outputs.pooler_output.squeeze().numpy())

    passage_embeddings = np.vstack(passage_embeddings)

    # Create FAISS index
    index = faiss.IndexFlatL2(passage_embeddings.shape[1])
    index.add(passage_embeddings)

    inputs = question_tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = question_encoder(**inputs)
    question_embedding = outputs.pooler_output.squeeze().numpy()

    distances, indices = index.search(np.array([question_embedding]), k)
    retrieved_passages = [corpus[idx] for idx in indices[0]]
    return retrieved_passages

# Example usage



if __name__ == '__main__':
    corpus = [
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials.",
    "The Statue of Liberty is a colossal neoclassical sculpture on Liberty Island in New York Harbor in New York, in the United States."
    ]
    question = "What is the Eiffel Tower?"
    retrieved_passages = getDPR(question, corpus)
    for i, passage in enumerate(retrieved_passages):
        print(f"Passage {i+1}: {passage}")