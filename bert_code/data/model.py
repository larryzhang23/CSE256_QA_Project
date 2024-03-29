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


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim, sent_length, kernel_size=7, num_filters=128, use_pad=True):
        """
        args:
            dim(int): glove_dim + char_dim
        """
        super().__init__()
        padding = "same" if use_pad else "valid"
        self.depth = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=False,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(dim, num_filters, kernel_size=1, bias=False)
        # self.dropout = nn.Dropout(p=0.1)
        self.layernorm = nn.LayerNorm([dim, sent_length], eps=1e-6)
        self.up = nn.Conv1d(dim, num_filters, kernel_size=1, bias=False)

    def forward(self, x):
        # x shape: [B, sen_length, dim]
        x_copy = self.up(x.permute(0, 2, 1))
        x = self.layernorm(x.permute((0, 2, 1)))  # [B, dim, sen_length]
        x = F.relu(self.pointwise(F.relu(self.depth(x))))
        return (x + x_copy).permute(0, 2, 1)  # [B, sen_length, dim]


class EmbeddingEncoder(nn.Module):
    def __init__(
        self, embedDim, sent_length, numFilters=128, numConvLayers=4, nHeads=8, ker_size=7,
    ):
        super().__init__()
        # convolution part
        conv = [
            DepthWiseConv1d(embedDim, sent_length=sent_length, num_filters=numFilters, kernel_size=ker_size)
        ]
        for _ in range(numConvLayers - 1):
            conv.append(
                DepthWiseConv1d(
                    numFilters, sent_length=sent_length, num_filters=numFilters
                )
            )
        self.conv = nn.Sequential(*conv)

        # transformer part
        self.transformerBlock = nn.TransformerEncoderLayer(
            numFilters,
            nhead=nHeads,
            dim_feedforward=numFilters * 4,
            layer_norm_eps=1e-6,
            norm_first=True,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transformerBlock(x)
        return x


class ContextQueryAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w0 = nn.Linear(in_features=dim * 3, out_features=1, bias=False)

    def forward(self, context, query):
        # context shape: [B, sent_length(400), 1, dim], query shape: [B, 1, sent_length(40), dim]
        contextSentLen, querySentLen = context.size(1), query.size(1)
        context = context.unsqueeze(2)
        query = query.unsqueeze(1)
        elemMul = context * query
        # simMat shape: [B, 400, 40, 3 * dim]
        simMat = torch.cat(
            [
                context.expand(-1, -1, querySentLen, -1),
                query.expand(-1, contextSentLen, -1, -1),
                elemMul,
            ],
            dim=-1,
        )
        # simMat shape: [B, 400, 40]
        simMat = self.w0(simMat)
        S = F.softmax(simMat, dim=-1).squeeze(-1)
        SS = F.softmax(simMat, dim=1).squeeze(-1)
        # out shape: [B, 400, dim]
        A = S @ query.squeeze(1)
        B = S @ SS.permute(0, 2, 1) @ context.squeeze(2)
        return A, B


class ModelEncoder(nn.Module):
    def __init__(
        self,
        embedDim,
        sent_length=400,
        numFilters=128,
        numConvLayers=2,
        nHeads=8,
        nBlocks=7,
    ):
        super().__init__()
        blocks = [
            EmbeddingEncoder(
                embedDim=embedDim,
                sent_length=sent_length,
                numFilters=numFilters,
                numConvLayers=numConvLayers,
                nHeads=nHeads,
                ker_size=5,
            )
        ]
        for _ in range(nBlocks - 1):
            blocks.append(
                EmbeddingEncoder(
                    embedDim=numFilters,
                    sent_length=sent_length,
                    numFilters=numFilters,
                    numConvLayers=numConvLayers,
                    nHeads=nHeads,
                    ker_size=5
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, C, A, B):
        concat = torch.cat([C, A, C * A, C * B], dim=-1)
        M0 = self.blocks(concat)
        M1 = self.blocks(M0)
        M2 = self.blocks(M1)
        return [M0, M1, M2]


class BaseClf(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb_q = InputEmbedding(numChar=numChar, dimChar=dimChar)
        self.input_emb_c = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=40
        )
        # [B, sent_length, 400]
        self.start_linear = nn.Linear(2 * (dimChar + dimGlove), 401)
        self.end_linear = nn.Linear(2 * (dimChar + dimGlove), 401)

    def forward(self, q, c):
        # [B, glove_dim + char_dim]
        emb_q = torch.mean(self.input_emb_q(q), dim=1)
        emb_c = torch.mean(self.input_emb_c(c), dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)


class BaseClf2(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb_q = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=40, dimGlove=dimGlove
        )
        self.input_emb_c = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=400, dimGlove=dimGlove
        )
        self.embed_enc_q = EmbeddingEncoder(dimChar + dimGlove, 40)
        self.embed_enc_c = EmbeddingEncoder(dimChar + dimGlove, 400)
        # [B, sent_length, 400]
        self.start_linear = nn.Linear(2 * (128), 401)
        self.end_linear = nn.Linear(2 * (128), 401)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.embed_enc_q(self.input_emb_q(q))
        emb_c = self.embed_enc_c(self.input_emb_c(c))
        emb_q = torch.mean(emb_q, dim=1)
        emb_c = torch.mean(emb_c, dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)


class BaseClf3(nn.Module):
    def __init__(self, numChar, dimChar=16, dimGlove=50) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb_q = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=40, dimGlove=dimGlove
        )
        self.input_emb_c = InputEmbedding(
            numChar=numChar, dimChar=dimChar, sent_length=400, dimGlove=dimGlove
        )
        self.embed_enc_q = EmbeddingEncoder(dimChar + dimGlove, 40)
        self.embed_enc_c = EmbeddingEncoder(dimChar + dimGlove, 400)
        self.context_query_attn = ContextQueryAttn(dim=128)
        self.model_enc = ModelEncoder(embedDim=4 * 128)
        # [B, sent_length, 400]

        self.start_linear = nn.Linear(2 * 128, 401)
        self.end_linear = nn.Linear(2 * 128, 401)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.embed_enc_q(self.input_emb_q(q))
        emb_c = self.embed_enc_c(self.input_emb_c(c))
        A, B = self.context_query_attn(emb_c, emb_q)
        outputs = self.model_enc(emb_c, A, B)
        emb_st = torch.cat((outputs[0], outputs[1]), dim=-1).mean(dim=1)
        emb_en = torch.cat((outputs[0], outputs[2]), dim=-1).mean(dim=1)
        return self.start_linear(emb_st), self.end_linear(emb_en)


if __name__ == "__main__":
    from dataset import SQuADQANet
    from torch.utils.data import DataLoader, Subset
    from model import InputEmbedding
    import torch.optim as optim
    import sys

    # sys.path.append("/Users/jwiroj/Desktop/CSE256_QA_Project/")
    sys.path.append("D:\\UCSD\\CSE256\\project")
    from trainer import trainer

    squadTrain = SQuADQANet("train")
    subsetTrain = Subset(squadTrain, [i for i in range(512)])
    # import pdb

    # pdb.set_trace()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = BaseClf3(numChar=squadTrain.charSetSize, dimChar=200, dimGlove=300)
    model.to(device)

    trainLoader = DataLoader(subsetTrain, batch_size=32, shuffle=True)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )
    criterion = nn.CrossEntropyLoss()

    # for epoch in range(2):
    #     for i, (contextDict, questionDict, label) in enumerate(trainLoader):
    #         print(epoch, i)
    #         model(contextDict, questionDict)
    #         quit()

    trainer(200, trainLoader, model, criterion, optimizer, device)
