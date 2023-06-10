import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
from torchtext.vocab import GloVe
from .preprocess import word_tokenize


class SQuADBase:

    """Wrapping class for hugging face dataset

    Args:
        split(str): train or validation
    """

    def __init__(self, split: str):
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


class SQuADQANet(SQuADBase, Dataset):

    """SQuAD dataset specialized for QANet

    Args:
        split(str): train or validation
        contextMaxLen(int): max length of the context
    """

    def __init__(self, split: str, contextMaxLen: int = 400, questionMaxLen: int = 40):
        super().__init__(split)
        print("Preparing Dataset...")
        self.legalDataIdx = []
        for i, sample in enumerate(self.dataset):
            if len(sample["context"]) <= contextMaxLen:
                self.legalDataIdx.append(i)
        self.contextMaxLen = contextMaxLen
        self.questionMaxLen = questionMaxLen
        self.glove = GloVe(name="6B", dim=50)
        self.char2idx = self._get_char2idx()
        self.idxHead = 0

    def __len__(self):
        return len(self.legalDataIdx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idxHead < len(self.legalDataIdx):
            idx = self.idxHead
            self.idxHead += 1
            return self[idx]

        raise StopIteration

    def _helper(self, item):
        # only support training format for now
        if item["answers"]["answer_start"]:
            startIdx = item["answers"]["answer_start"][0]
            text = item["answers"]["text"][0]
            item["answers"]["text"] = text
            item["answers"]["index"] = (startIdx, startIdx + len(text) - 1)
            item["answers"].pop("answer_start")
        else:
            # for unanswerable questions, set startIdx = 400, endIdx = 400
            item["answers"]["text"] = ""
            item["answers"]["index"] = (self.contextMaxLen, self.contextMaxLen)
            item["answers"].pop("answer_start")
        return item

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
        for elemIdx in self.legalDataIdx:
            article = self.dataset[elemIdx]
            context, question = article["context"], article["question"]
            context, question = context.lower(), question.lower()
            context = context.replace("''", '" ').replace("``", '" ')
            question = question.replace("''", '" ').replace("``", '" ')
            charSet.update(question)
            charSet.update(context)
        char2idx = {c: idx for idx, c in enumerate(charSet)}
        vocab_size = len(char2idx)
        char2idx["unk"] = vocab_size
        char2idx["pad"] = vocab_size + 1
        return char2idx

    def __getitem__(self, idx):
        elemIdx = self.legalDataIdx[idx]
        item = self.dataset[elemIdx]
        item = self._helper(item)
        contextDict = self._get_embedding_idx(
            item["context"], sent_length=self.contextMaxLen
        )
        questionDict = self._get_embedding_idx(
            item["question"], sent_length=self.questionMaxLen
        )
        index = torch.tensor(item["answers"]["index"], dtype=torch.int64)
        return contextDict, questionDict, index

    @property
    def charSetSize(self):
        return len(self.char2idx)


class SQuADBert(SQuADBase):
    """Wrapping class for hugging face dataset

    Args:
        split(str): train or validation
        model_name(str): name of the model in huggingface
    """

    def __init__(self, split: str, model_name: str):
        super().__init__(split)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenized_dataset = self._preprocess()

    def _preprocess(self):
        is_train = "train" in self.split

        def dataset_transform(dataset):
            questions = dataset["question"]
            questions = [q.strip() for q in questions]
            contexts = dataset["context"]
            inputs = self.tokenizer(
                questions,
                contexts,
                max_length=400,
                truncation="only_second",
                return_offsets_mapping=True,
                return_overflowing_tokens=not is_train,
                padding="max_length",
            )

            offset_mapping = None
            if is_train:
                offset_mapping = inputs.pop("offset_mapping")
            sample_map = None
            if not is_train:
                sample_map = inputs.pop("overflow_to_sample_mapping")
            answers = dataset["answers"]
            start_positions = []
            end_positions = []
            example_ids = []

            if is_train:
                for i, offset in enumerate(offset_mapping):
                    answer = answers[i]
                    if len(answer["answer_start"]) == 0:
                        start_positions.append(0)
                        end_positions.append(0)
                        continue
                    start_char = answer["answer_start"][0]
                    end_char = answer["answer_start"][0] + len(answer["text"][0])
                    sequence_ids = inputs.sequence_ids(i)

                    idx = 0
                    while sequence_ids[idx] != 1:
                        idx += 1
                    context_start = idx
                    while sequence_ids[idx] == 1:
                        idx += 1
                    context_end = idx - 1

                    if (
                        offset[context_start][0] > end_char
                        or offset[context_end][1] < start_char
                    ):
                        start_positions.append(0)
                        end_positions.append(0)
                    else:
                        idx = context_start
                        while idx <= context_end and offset[idx][0] <= start_char:
                            idx += 1
                        start_positions.append(idx - 1)

                        idx = context_end
                        while idx >= context_start and offset[idx][1] >= end_char:
                            idx -= 1
                        end_positions.append(idx + 1)
            else:
                for i in range(len(inputs["input_ids"])):
                    sample_idx = sample_map[i]
                    example_ids.append(dataset["id"][sample_idx])

                    sequence_ids = inputs.sequence_ids(i)
                    offset = inputs["offset_mapping"][i]
                    inputs["offset_mapping"][i] = [o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)]

            if is_train:
                inputs["start_positions"] = start_positions
                inputs["end_positions"] = end_positions
            else:
                inputs["example_id"] = example_ids
                pass
            return inputs

        tokenized_dataset = self.dataset.map(dataset_transform, batched=True, remove_columns=self.dataset.column_names)
        # tokenized_dataset = tokenized_dataset.remove_columns(["id", "title", "context", "question", "answers"])
        if is_train:
            tokenized_dataset.set_format("torch")
        return tokenized_dataset

    def __iter__(self):
        return iter(self.tokenized_dataset)

    def __getitem__(self, idx):
        return self.tokenized_dataset[idx]

    def __len__(self):
        return len(self.tokenized_dataset)

    def __str__(self):
        return str(self.tokenized_dataset)


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
