from simcse import SimCSE


def get_top(documents, query, k=5, is_embed=False):

    model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

    #if not is_embed:
        #embeddings=[model.encode(phrase) for phrase in documents]
    model.build_index(documents)

    results = model.search(query)
    return results[:min(k, len(results))]
