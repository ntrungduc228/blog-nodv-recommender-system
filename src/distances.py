import numpy as np
from scipy.stats import entropy
from gensim import similarities
from src import settings

def jensen_shannon(query, matrix):
    """
    This function implements a Jensen-Shannon similarity
    between the input query (an LDA topic distribution for a document)
    and the entire corpus of topic distributions.
    It returns an array of length M (the number of documents in the corpus)
    """
    # lets keep with the p,q notation above
    p = query[None, :].T  # take transpose
    q = matrix.T  # transpose matrix

    m = 0.5 * (p + q)
    return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))


def get_most_similar_documents(query, matrix, k=7):
    """
    This function implements the Jensen-Shannon distance above
    and returns the top k indices of the smallest jensen shannon distances
    """
    # list of jensen shannon distances
    sims = jensen_shannon(query, matrix)
    # the top k positional index of the smallest Jensen Shannon distances

    # return sims.argsort()[:k]
    print('sims ', sims[sims.argsort()])
    print('=======================\n')

    print('list ', list(sims[sims.argsort()])[:k])
    print('list index ', list(sims.argsort())[:k], '\n')

    print('list reverse', list(sims[sims.argsort()])[::-1][:k])
    print('list reverse index', list(sims.argsort())[::-1][:k])
    arr = sims.argsort()
    return arr[:k]


index = similarities.MatrixSimilarity.load(settings.PATH_MATRIX_SIMILARITY)
def get_posts_similarity(vector_doc, k=10):
    sims = index[vector_doc]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    arr = [tup[0] for tup in sims[:k]]
    return arr