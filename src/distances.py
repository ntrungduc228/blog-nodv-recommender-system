import numpy as np
from scipy.stats import entropy
from gensim import similarities
import settings


index = similarities.MatrixSimilarity.load(settings.PATH_MATRIX_SIMILARITY)

def get_posts_similarity(vector_doc, k=100):
    sims = index[vector_doc]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    arr = [tup[0] for tup in sims[:k]]
    return arr