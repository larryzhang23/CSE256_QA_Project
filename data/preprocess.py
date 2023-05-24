from tqdm import tqdm
import spacy

nlp = spacy.blank("en")
from torchtext.vocab import GloVe
import torch
import torch.nn as nn

"""
FROM PAPER
glove_dim = 300
char_limit = 16
char_dim = 200
"""

"""
FOR DEV
glove_dim = 50
char_limit = 16
char_dim = 32
"""


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


CONTEXT_MAX_LEN = 400


def get_vocabs(wordSet=set(), charSet=set(), articles=None):
    """
    wordSet: set of all possible words
    charSet: set of all possible characters
    articles:
    """

    # We will be using GloVe uncased
    # Need to lower case every word/char

    for it, article in enumerate(tqdm(articles)):
        if len(article["context"]) >= CONTEXT_MAX_LEN:
            continue
        context, question = article["context"], article["question"]
        context, question = context.lower(), question.lower()
        context = context.replace("''", '" ').replace("``", '" ')
        question = question.replace("''", '" ').replace("``", '" ')

        tokenized_context = word_tokenize(context)
        tokenized_question = word_tokenize(question)

        wordSet.update(tokenized_context)
        wordSet.update(tokenized_question)
        charSet.update(context)
        charSet.update(question)

    return wordSet, charSet


def get_word2idx_embedding(wordSet=None, name="6B", dim=50):
    """Create word2idx dict
    word2idx is a dict from our wordSet and some special tokens to
    the index of the embedding matrix.
    args:
        wordSet: All possible words in corpus
        dim (int): dimension of GloVe
    """
    glove = GloVe(name=name, dim=dim)
    special_tokens = ["unk", "pad"]

    word2idx = {}

    idx = 0
    for word in glove.stoi.keys():
        if word in wordSet:
            word2idx[word] = idx
            idx += 1

    numNormalWords = len(word2idx.keys())

    for i, word in special_tokens:
        word2idx[word] = numNormalWords + i

    vocab_size = len(word2idx.keys())
    custom_embedding = torch.zeros((vocab_size, dim))

    for i, word in enumerate(special_tokens):
        # Glove has 'unk' and 'pad' vectors already
        custom_embedding[i + numNormalWords] = glove.vectors[glove.stoi[word]]
        # Can also try: torch.randn(dim)

    for word, idx in word2idx.items():
        custom_embedding[idx] = glove.vectors[glove.stoi[word]]

    embedding_layer = nn.Embedding.from_pretrained(custom_embedding)
    # Parameters are stored in embedding_layer.weight
    return word2idx, embedding_layer


def get_char2idx_embedding(charSet, dim=32):
    char2idx = {c: idx for idx, c in enumerate(charSet)}
    vocab_size = len(char2idx.keys())
    char2idx["unk"] = vocab_size
    char2idx["pad"] = vocab_size + 1
    vocab_size = len(char2idx.keys())

    emb_matrix = nn.Embedding(vocab_size, dim)
    # May need to initalize weights here
    return char2idx, emb_matrix


if __name__ == "__main__":
    charSet = set()
    charSet.update("abs")
    print(charSet)
