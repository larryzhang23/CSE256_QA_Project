import torch
import torch.nn as nn
from torchtext.vocab import GloVe
import torch.nn.functional as F


class HighwayNetwork(nn.Module):
    def __init__(self, dim, num_layers=2, act_type="relu"):
        super().__init__()
        self.n_layers = num_layers
        self.act = nn.ReLU() if act_type == "relu" else nn.Sigmoid()
        self.linear_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )
        self.gate_layers = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(num_layers)]
        )

    def forward(self, x):
        for i in range(self.n_layers):
            T = F.sigmoid(self.gate_layers[i](x))
            H = self.act(self.linear_layers[i](x))
            x = T * H + (1 - T) * x
        return x


class InputEmbedding(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50):
        super().__init__()
        self.charEmbed = nn.Embedding(numChar, dimChar)
        glove = GloVe(name="6B", dim=dimGlove)
        self.gloveEmbed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        self.conv = nn.Conv2d(dimChar, dimChar, (1, 5))
        self.hn = HighwayNetwork(dim=dimChar)
        # self.unkIdx = glove.stoi["unk"]
        # self.unk = nn.Parameter(glove["unk"])

    def forward(self, x):
        # wordIdxTensor shape: [B, sent_length], charIdxTensor shape: [B, sent_length, 16]
        wordIdxTensor, charIdxTensor = x["wordIdx"], x["charIdx"]

        # charEmbedding shape: [B, sent_length, 16, char_dim]
        charEmbed = self.charEmbed(charIdxTensor)
        charEmbed = self.conv(charEmbed.permute(0, 3, 1, 2))
        charEmbed, _ = torch.max(charEmbed, dim=-1)
        charEmbed = charEmbed.permute(0, 2, 1)  # new shape: [B,sent_length, char_dim]

        # wordEmbedding shape: [B, sent_length, glove_dim]
        wordEmbed = self.gloveEmbed(wordIdxTensor)
        # import pdb; pdb.set_trace()

        catEmbed = torch.concat((wordEmbed, charEmbed), dim=2)
        catEmbed = self.hn(catEmbed)
        # mask = wordIdxTensor == self.unkIdx

        # if wordEmbed[mask].shape[0]:
        #     wordEmbed[mask] = 0
        #     wordEmbed[mask] += self.unk
