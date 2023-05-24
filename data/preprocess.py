from tqdm import tqdm
import spacy
nlp = spacy.blank("en")

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

if __name__ == "__main__":
    charSet = set()
    charSet.update("abs")
    print(charSet)


