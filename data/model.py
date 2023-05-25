import torch 
import torch.nn as nn
from torchtext.vocab import GloVe

class InputEmbedding(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50):
        super().__init__()
        self.charEmbed = nn.Embedding(numChar, dimChar)
        glove = GloVe(name="6B", dim=dimGlove)
        self.gloveEmbed = nn.Embedding.from_pretrained(glove.vectors, freeze=True)
        # self.unkIdx = glove.stoi["unk"]
        # self.unk = nn.Parameter(glove["unk"])

        
    def forward(self, x):
        # wordIdxTensor shape: [B, sent_length], charIdxTensor shape: [B, sent_length, 16]
        wordIdxTensor, charIdxTensor = x["wordIdx"], x["charIdx"]

        # wordEmbedding shape: [B, sent_length, glove_dim], charEmbedding shape: [B, sent_length, 16, char_dim]
        charEmbed = self.charEmbed(charIdxTensor)
        wordEmbed = self.gloveEmbed(wordIdxTensor)
        # import pdb; pdb.set_trace()
        
        # mask = wordIdxTensor == self.unkIdx
        
        # if wordEmbed[mask].shape[0]:
        #     wordEmbed[mask] = 0
        #     wordEmbed[mask] += self.unk 
        