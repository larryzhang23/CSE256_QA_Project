from tqdm import tqdm
import spacy

nlp = spacy.blank("en")
from torchtext.vocab import GloVe
import torch
import torch.nn as nn


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
    for it, article in enumerate(tqdm(articles)):
        if len(article["context"]) >= CONTEXT_MAX_LEN:
            continue
        context, question = article["context"], article["question"]
        context = context.replace("''", '" ').replace("``", '" ')
        question = question.replace("''", '" ').replace("``", '" ')

        tokenized_context = word_tokenize(context)
        tokenized_question = word_tokenize(question)

        wordSet.update(tokenized_context)
        wordSet.update(tokenized_question)
        charSet.update(context)
        charSet.update(question)

    return wordSet, charSet


def get_word2idx_embedding(wordSet=None, dim=50):
    """Create word2idx dict
    word2idx is a dict from our wordSet and some special tokens to
    the index of the embedding matrix.
    args:
        wordSet: All possible words in corpus
        dim (int): dimension of GloVe
    """
    glove = GloVe(name="6B", dim=dim)
    special_tokens = ["<UNK>", "<PAD>"]

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

    for idx, token in enumerate(special_tokens):
        custom_embedding[idx + numNormalWords] = torch.randn(dim)

    for word, idx in word2idx.items():
        custom_embedding[idx] = glove.vectors[glove.stoi[word]]

    embedding_layer = nn.Embedding.from_pretrained(custom_embedding)
    # Parameters are stored in embedding_layer.weight
    return word2idx, embedding_layer


# def get_char2idx_embedding(charSet,dim=16):

if __name__ == "__main__":
    charSet = set()
    charSet.update("abs")
    print(charSet)
