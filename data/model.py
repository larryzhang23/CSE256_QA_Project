import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    def __init__(self, sent_length, num_layers=2, act_type="relu"):
        super().__init__()
        self.n_layers = num_layers
        self.act = nn.ReLU() if act_type == "relu" else nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(sent_length, sent_length) for _ in range(num_layers)]
        )
        self.gate_layers = nn.ModuleList(
            [nn.Linear(sent_length, sent_length) for _ in range(num_layers)]
        )

    def forward(self, x):
        # x: [B, sent_length, dim]
        x = x.permute((0, 2, 1))  # [B, dim, sent_length]

        for i in range(self.n_layers):
            T = F.sigmoid(self.gate_layers[i](x))
            H = self.act(self.linear_layers[i](x))
            x = T * H + (1 - T) * x
        return x.permute((0, 2, 1))  # [B, sent_length, dim]


class InputEmbedding(nn.Module):
    def __init__(self, numChar, sent_length=400, dimChar=16, dimGlove=50):
        super().__init__()
        self.charEmbed = nn.Embedding(numChar, dimChar)
        glove = GloVe(name="6B", dim=dimGlove)
        self.gloveEmbed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        self.conv = nn.Conv2d(dimChar, dimChar, (1, 5))
        self.hn = HighwayNetwork(sent_length)

    def forward(self, x):
        # wordIdxTensor shape: [B, sent_length], charIdxTensor shape: [B, sent_length, 16]
        wordIdxTensor, charIdxTensor = x["wordIdx"], x["charIdx"]
        # charEmbedding shape: [B, sent_length, 16, char_dim]
        charEmbed = self.charEmbed(charIdxTensor)
        charEmbed = self.conv(charEmbed.permute(0, 3, 1, 2))
        charEmbed, _ = torch.max(charEmbed, dim=-1)
        charEmbed = charEmbed.permute(0, 2, 1)  # new shape: [B,sent_length, char_ndim]
        # wordEmbedding shape: [B, sent_length, glove_dim]
        wordEmbed = self.gloveEmbed(wordIdxTensor)
        # import pdb; pdb.set_trace()
        catEmbed = torch.cat(
            (wordEmbed, charEmbed), dim=2
        )  # [B, sent_length, glove_dim + char_dim]
        catEmbed = self.hn(catEmbed)
        return catEmbed


class BaseClf(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb_q = InputEmbedding(numChar=numChar, dimChar=dimChar)
        self.input_emb_c = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=40
        )
        # [B, sent_length, 400]
        self.start_linear = nn.Linear(2 * (dimGlove + dimChar), 401)
        self.end_linear = nn.Linear(2 * (dimGlove + dimChar), 401)

    def forward(self, q, c):
        # [B, glove_dim + char_dim]

        emb_q = torch.mean(self.input_emb_q(q), dim=1)
        emb_c = torch.mean(self.input_emb_c(c), dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)


if __name__ == "__main__":
    from dataset import SQuADQANet
    from torch.utils.data import DataLoader, Subset
    from model import InputEmbedding
    import torch.optim as optim
    import sys

    sys.path.append("/Users/jwiroj/Desktop/CSE256_QA_Project/")
    from trainer import trainer

    squadTrain = SQuADQANet("train")
    subsetTrain = Subset(squadTrain, [i for i in range(128)])
    # import pdb

    # pdb.set_trace()
    model = BaseClf(numChar=squadTrain.charSetSize)

    trainLoader = DataLoader(subsetTrain, batch_size=32, shuffle=False)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-3,
    )
    criterion = nn.CrossEntropyLoss()

    # for epoch in range(2):
    #     for i, (contextDict, questionDict, label) in enumerate(trainLoader):
    #         print(epoch, i)
    #         model(contextDict, questionDict)
    #         quit()

    trainer(50, trainLoader, model, criterion, optimizer)
