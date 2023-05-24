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
        if len(article["context"]) >= CONTEXT_MAX_LEN: continue
        context,question = article["context"], article['question']
        context  = context.replace("''", '" ').replace("``", '" ')
        question = question.replace("''", '" ').replace("``", '" ')

        tokenized_context = word_tokenize(context)
        tokenized_question = word_tokenize(question)

        wordSet.update(tokenized_context)
        wordSet.update(tokenized_question)
        charSet.update(context)
        charSet.update(question)

    return wordSet, charSet

def get_word2idx_embedding(wordSet=None, dim=50):
    """ Create word2idx dict
    word2idx is a dict from our wordSet to the index of the Glove Embedding
    args:
        wordSet
        dim (int): dimension of GloVe
    """
    glove = GloVe(name='6B', dim=dim)
    special_tokens = ["<UNK>","<PAD>"]

    word2idx = {word: i for i, word in enumerate(glove.itos) if word in wordSet}

    numNormalWords = len(word2idx.keys())
    for i,word in special_tokens:
        word2idx[word] = numNormalWords+i

    vocab_size = len(word2idx.keys()) + len(special_tokens)
    custom_embedding = torch.zeros((vocab_size,dim))

    for i, token in enumerate(special_tokens):
        custom_embedding[i + numNormalWords] = torch.randn(dim)

    for word,idx in word2idx.items():
        custom_embedding[idx] = glove.vectors[glove.stoi[word]]
    
    embedding_layer = nn.Embedding.from_pretrained(custom_embedding)
    # Freeze everything up to special token
    embedding_layer.weight[:numNormalWords].requires_grad = False

    # with open(emb_file,'r') as f:
    #     emb_dict = {}
    #     for line in f:
    #         # elem[0] is the word
    #         # elem[1:] is the embedding
    #         elem = line.split()
    #         word,emb = elem[0], elem[1:]
    #         if word in wordSet:
    #             emb_dict[word] = emb
    #     word2ind = {token:idx for idx,token in enumerate(emb_dict.keys())}
    return word2idx, embedding_layer

if __name__ == "__main__":
    charSet = set()
    charSet.update("abs")
    print(charSet)


