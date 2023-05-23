from datasets import load_dataset

class SQuADBase:

    """ Wrapping class for hugging face dataset

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


class SQuADQANet(SQuADBase):
    
    """ SQuAD dataset specialized for QANet

    Args:
        split(str): train or validation
        contextMaxLen(int): max length of the context
    """
    def __init__(self, split: str, contextMaxLen: int = 400):
        super().__init__(split)
        self.legalDataIdx = []
        self.contextMaxLen = contextMaxLen
        for i, sample in enumerate(self.dataset):
            if len(sample["context"]) <= contextMaxLen:
                self.legalDataIdx.append(i)
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
    
    def __getitem__(self, idx):
        elemIdx = self.legalDataIdx[idx]
        item = self.dataset[elemIdx]
        item = self._helper(item)
        return item

# Test code
if __name__ == '__main__':
    squadTrain = SQuADQANet("train")
    print(len(squadTrain))
    # print(squadTrain[0])
    for i, sample in enumerate(squadTrain):
        print(sample)
        if i >= 5:
            break
    
    
