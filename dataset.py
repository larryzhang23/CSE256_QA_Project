import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchtext.vocab import GloVe
import spacy
nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]

class SQuADBase:

    """Wrapping class for hugging face dataset

    Args:
        split(str): train or validation
    """

    def __init__(self, version, split: str):
        if version == "v1":
            self.dataset = load_dataset("squad", split=split)
        else:
            self.dataset = load_dataset("squad_v2", split=split)
        self.split = split

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def __str__(self):
        return str(self.dataset)


class SQuADQANet(Dataset):

    """SQuAD dataset specialized for QANet

    Args:
        split(str): train or validation
        questionMaxLen(int): max length of the question
        contextMaxLen(int): max length of the context
        version(str): dataset version (v1 or v2)
        glove_version: glove embedding version
        glove_dim: glove embedding dimension
    """

    def __init__(
            self, 
            questionMaxLen: int = 50, 
            contextMaxLen: int = 400, 
            version: str = "v1", 
            glove_version: str = "6B", 
            glove_dim=300
        ):
        self.contextMaxLen = contextMaxLen
        self.questionMaxLen = questionMaxLen
        self.glove = GloVe(name=glove_version, dim=glove_dim)
        self.trainLegalDataIdx = []
        self.valLegalDataIdx = []
        self.train_dataset = SQuADBase(version, "train")
        print("Preparing train dataset...")
        self._helper("train")
        self.char2idx = self._get_char2idx()

        print("Preparaing val dataset...")
        self.val_dataset = SQuADBase(version, "validation")
        self._helper("validation")
        self.split = "train"
       

    def __len__(self):
        if self.split == "train":
            return len(self.trainLegalDataIdx)
        else:
            return len(self.valLegalDataIdx)

    def _helper(self, split):
        if split == "train":
            self.train_index = []
            self.train_spans = []
            this_index = self.train_index 
            this_spans = self.train_spans
            dataset = self.train_dataset
            legalIdx = self.trainLegalDataIdx
        else:
            self.val_index = []
            self.val_spans = []
            this_index = self.val_index 
            this_spans = self.val_spans
            dataset = self.val_dataset
            legalIdx = self.valLegalDataIdx

        for i, sample in enumerate(dataset):
            removeIdx = []
            for i, answer in enumerate(sample["answers"]["text"]):
                answer = answer.lower().replace("''", '" ').replace("``", '" ')
                ansTokens = word_tokenize(answer)
                if len(ansTokens) > self.questionMaxLen:
                    removeIdx.append(i)
            context = sample["context"].lower().replace("''", '" ').replace("``", '" ')
            tokens = word_tokenize(context)
            if len(tokens) > self.contextMaxLen or len(removeIdx) == len(sample["answers"]["text"]):
                continue
            for idx in removeIdx:
                sample["answers"]["text"].pop(idx)
                sample["answers"]["answer_start"].pop(idx)
            legalIdx.append(i)
            spans = self._convert_idx(context, tokens)
            ansIdx = []
            for text, startIdx in zip(sample["answers"]["text"], sample["answers"]["answer_start"]):
                endIdx = startIdx + len(text)
                answerTokenids = []
                for idx, span in enumerate(spans):
                    if endIdx <= span[0] or startIdx >= span[1]:
                        continue
                    answerTokenids.append(idx)
                if answerTokenids:
                    ansIdx.append((answerTokenids[0], answerTokenids[-1]))
                else:
                    print(sample)
                    print(spans)
                    print(startIdx, endIdx)
                    print(tokens)
                    raise Exception("error")
            this_spans.append(spans)
            this_index.append(ansIdx)

    def _get_embedding_idx(self, sent, word_length=16, sent_length=400):
        tokens = word_tokenize(sent.lower().replace("''", '" ').replace("``", '" '))
        pad_idx = self.glove.stoi["pad"]
        unk_idx = self.glove.stoi["unk"]
        char_pad_idx = self.char2idx["pad"]
        char_unk_idx = self.char2idx["unk"]
        res = np.ones(sent_length) * pad_idx
        char_res = np.ones((sent_length, word_length)) * char_pad_idx
        for i, token in enumerate(tokens):
            if token in self.glove.stoi:
                res[i] = self.glove.stoi[token]
            else:
                res[i] = unk_idx

            for j, ch in enumerate(token):
                if j >= word_length:
                    break
                if ch in self.char2idx:
                    char_res[i, j] = self.char2idx[ch]
                else:
                    char_res[i, j] = char_unk_idx

        return {
            "wordIdx": torch.tensor(res, dtype=torch.int64),
            "charIdx": torch.tensor(char_res, dtype=torch.int64),
        }

    def _get_char2idx(self):
        charSet = set()
        for elemIdx in self.trainLegalDataIdx:
            article = self.train_dataset[elemIdx]
            context, question = article["context"], article["question"]
            context = context.lower().replace("''", '" ').replace("``", '" ')
            question = question.lower().replace("''", '" ').replace("``", '" ')
            contextTokens = word_tokenize(context)
            questionTokens = word_tokenize(question)
            for token in contextTokens:
                charSet.update(token)
            for token in questionTokens:
                charSet.update(token)
        char2idx = {c: idx for idx, c in enumerate(charSet)}
        vocab_size = len(char2idx)
        char2idx["unk"] = vocab_size
        char2idx["pad"] = vocab_size + 1
        return char2idx

    def _convert_idx(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans
    
    def __getitem__(self, idx):
        if self.split == "train":
            elemIdx = self.trainLegalDataIdx[idx]
            item = self.train_dataset[elemIdx]
            index_set = self.train_index
        else:
            elemIdx = self.valLegalDataIdx[idx]
            item = self.val_dataset[elemIdx]
            index_set = self.val_index
        contextDict = self._get_embedding_idx(
            item["context"], sent_length=self.contextMaxLen
        )
        questionDict = self._get_embedding_idx(
            item["question"], sent_length=self.questionMaxLen
        )
        index = index_set[idx]
        if self.split == "validation":
            if 0 < len(index) < 6:
                index.extend([index[0] for _ in range(6 - len(index))])
        else:
            index = index[0]
        index = torch.tensor(index, dtype=torch.int64)
        return contextDict, questionDict, index

    @property
    def charSetSize(self):
        return len(self.char2idx)
    
    def setSplit(self, split):
        assert split in ["train", "validation"]
        self.split = split
    
    def getSampleMeta(self, idx):
        elemIdx = self.legalDataIdx[idx]
        item = self.dataset[elemIdx]
        return item["context"], item["spans"], item["id"]


# Test code
if __name__ == "__main__":
    # build actual dataset
    from torch.utils.data import DataLoader
    from model import InputEmbedding

    squadTrain = SQuADQANet("train")
    model = InputEmbedding(squadTrain.charSetSize)
    # print(len(squadTrain))
    # print(squadTrain[0])
    # for i, sample in enumerate(squadTrain):
    #     print(sample)
    #     if i >= 5:
    #         break
    trainLoader = DataLoader(squadTrain, batch_size=32, shuffle=True)
    for epoch in range(2):
        for i, (contextDict, questionDict, label) in enumerate(trainLoader):
            print(epoch, i)
            model(contextDict)
