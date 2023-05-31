import math
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
        # x: [B, sent_length, dim]
        for i in range(self.n_layers):
            T = F.sigmoid(self.gate_layers[i](x))
            H = self.act(self.linear_layers[i](x))
            x = T * H + (1 - T) * x
        return x  # [B, sent_length, dim]


class InputEmbedding(nn.Module):
    def __init__(self, numChar, dimChar=200, dimGlove=300, freeze=True):
        super().__init__()
        self.charEmbed = nn.Embedding(numChar, dimChar)
        glove = GloVe(name="6B", dim=dimGlove)
        self.wordPadIdx = glove.stoi["pad"]
        self.charPadIdx = numChar - 1
        self.gloveEmbed = nn.Embedding.from_pretrained(glove.vectors, freeze=freeze)
        self.conv = nn.Conv2d(dimChar, dimChar, (1, 5))
        self.hn = HighwayNetwork(dimChar + dimGlove)

    def forward(self, x):
        # wordIdxTensor shape: [B, sent_length], charIdxTensor shape: [B, sent_length, 16]
        wordIdxTensor, charIdxTensor = x["wordIdx"], x["charIdx"]
        # charEmbedding shape: [B, sent_length, 16, char_dim]
        charEmbed = self.charEmbed(charIdxTensor)
        charEmbed = F.relu(self.conv(charEmbed.permute(0, 3, 1, 2)))
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
    def __init__(self, dim, kernel_size=7, num_filters=128, use_pad=True):
        """
        args:
            dim(int): glove_dim + char_dim
        """
        super().__init__()
        padding = "same" if use_pad else "valid"
        self.depth = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            groups=num_filters,
            bias=False,
            padding=padding,
        )
        self.pointwise = nn.Conv1d(num_filters, num_filters, kernel_size=1, bias=False)
        self.map = nn.Conv1d(dim, num_filters, kernel_size=1, bias=False)
        # self.dropout = nn.Dropout(p=0.1)
        self.layernorm = nn.LayerNorm(num_filters)

    def forward(self, x):
        # x shape: [B, sen_length, dim]
        x = x.permute(0, 2, 1)
        x = self.map(x)
        x_res = x
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = F.relu(self.pointwise(F.relu(self.depth(x))))
        return (x + x_res).permute(0, 2, 1)  # [B, sen_length, dim]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, sent_length, embedding_dim]``
        """
        return x + self.pe


class EmbeddingEncoder(nn.Module):
    def __init__(
        self,
        embedDim,
        numFilters=128,
        numConvLayers=4,
        nHeads=8,
        ker_size=7,
    ):
        super().__init__()
        # convolution part
        conv = [
            DepthWiseConv1d(
                embedDim,
                num_filters=numFilters,
                kernel_size=ker_size,
            )
        ]
        for _ in range(numConvLayers - 1):
            conv.append(DepthWiseConv1d(numFilters, num_filters=numFilters))
        self.conv = nn.Sequential(*conv)

        # transformer part
        self.transformerBlock = nn.TransformerEncoderLayer(
            numFilters,
            nhead=nHeads,
            dim_feedforward=numFilters * 4,
            norm_first=True,
            batch_first=True,
            dropout=0.0,
        )
        self.nHeads = nHeads

    def forward(self, x, mask_idx=None):
        x = self.conv(x)
        sent_length = x.size(1)
        if mask_idx is not None:
            mask = torch.logical_or(mask_idx.unsqueeze(-1).expand(-1, -1, sent_length), mask_idx.unsqueeze(1).expand(-1, sent_length, -1))
            mask = mask.unsqueeze(1).expand(-1, self.nHeads, -1, -1).reshape((mask.size(0) * self.nHeads, sent_length, sent_length))
        else:
            mask = None
        x = self.transformerBlock(x, mask)
        if mask_idx is not None:
            x = torch.nan_to_num(x)
        return x


class ContextQueryAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w0 = nn.Linear(in_features=dim * 3, out_features=1, bias=False)

    def forward(self, context, query, context_mask=None, query_mask=None):
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
        simMat = self.w0(simMat).squeeze(-1)
        if context_mask is not None and query_mask is not None:
            mask = torch.logical_or(context_mask.unsqueeze(-1).expand(-1, -1, querySentLen), query_mask.unsqueeze(1).expand(-1, contextSentLen, -1))
            simMat = simMat.masked_fill(mask, -1e30)
            
        S = F.softmax(simMat, dim=-1)
        SS = F.softmax(simMat, dim=1)
        if context_mask is not None and query_mask is not None:
            S = torch.nan_to_num(S)
            SS = torch.nan_to_num(SS)
        SS = torch.nan_to_num(SS)
        # out shape: [B, 400, dim]
        A = S @ query.squeeze(1)
        B = S @ SS.permute(0, 2, 1) @ context.squeeze(2)
        return A, B


class ModelEncoder(nn.Module):
    def __init__(
        self,
        embedDim,
        numConvLayers=2,
        nHeads=8,
        nBlocks=7,
    ):
        super().__init__()
        self.embedDim = embedDim
        blocks = []
        for _ in range(nBlocks):
            blocks.append(
                EmbeddingEncoder(
                    embedDim=embedDim,
                    numFilters=embedDim,
                    numConvLayers=numConvLayers,
                    nHeads=nHeads,
                    ker_size=5,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(embedDim * 4, embedDim, bias=False)

    def forward(self, C, A, B, mask_idx=None):
        concat = torch.cat([C, A, C * A, C * B], dim=-1)
        concat = self.linear(concat)
        M0 = concat 
        for block in self.blocks:
            M0 = block(M0, mask_idx)
        M1 = M0
        for block in self.blocks:
            M1 = block(M1, mask_idx)
        M2 = M1 
        for block in self.blocks:
            M2 = block(M2, mask_idx)
        return [M0, M1, M2]


class ModelEncoderV2(nn.Module):
    def __init__(
        self,
        embedDim,
        numConvLayers=2,
        nHeads=8,
        nBlocks=7,
    ):
        super().__init__()
        self.embedDim = embedDim
        blocks = []
        for _ in range(nBlocks):
            blocks.append(
                EmbeddingEncoder(
                    embedDim=embedDim,
                    numFilters=embedDim,
                    numConvLayers=numConvLayers,
                    nHeads=nHeads,
                    ker_size=5,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(embedDim * 4, embedDim, bias=False)

    def forward(self, C, A, B):
        concat = torch.cat([C, A, C * A, C * B], dim=-1)
        concat = self.linear(concat)
        M0 = self.blocks(concat)
        return M0


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
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove
        )
        self.embed_enc = EmbeddingEncoder(dimChar + dimGlove)
        # [B, sent_length, 400]
        self.start_linear = nn.Linear(2 * (128), 401)
        self.end_linear = nn.Linear(2 * (128), 401)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.embed_enc(self.input_emb(q))
        emb_c = self.embed_enc(self.input_emb(c))
        emb_q = torch.mean(emb_q, dim=1)
        emb_c = torch.mean(emb_c, dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)

    def count_params(self):
        num_params = sum(param.numel() for param in self.parameters())
        return f"{(num_params / 1e6):.2f}M"


class QANet(nn.Module):
    def __init__(
        self, numChar, dim=128, dimChar=200, dimGlove=300, freeze=True
    ) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, freeze=freeze
        )

        self.embed_enc = EmbeddingEncoder(dimChar + dimGlove)
        self.context_query_attn = ContextQueryAttn(dim=dim)
        self.model_enc = ModelEncoder(embedDim=dim)
        # [B, sent_length, 401]
        self.start_linear = nn.Linear(2 * dim, 1)
        self.end_linear = nn.Linear(2 * dim, 1)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        c_word_idx = c["wordIdx"]
        q_word_idx = q["wordIdx"]
        word_pad_idx = self.input_emb.wordPadIdx
        # c_word_mask = c_word_idx == word_pad_idx 
        # q_word_mask = q_word_idx == word_pad_idx 
        c_word_mask, q_word_mask = None, None
        emb_q = self.embed_enc(self.input_emb(q), q_word_mask)
        emb_c = self.embed_enc(self.input_emb(c), c_word_mask)
        A, B = self.context_query_attn(emb_c, emb_q, c_word_mask, q_word_mask)
        outputs = self.model_enc(emb_c, A, B, c_word_mask)
        emb_st = torch.cat((outputs[0], outputs[1]), dim=-1)
        emb_en = torch.cat((outputs[0], outputs[2]), dim=-1)
        # pred_start = self.start_linear(emb_st).squeeze().masked_fill(c_word_mask, -1e30)
        # pred_end = self.end_linear(emb_en).squeeze().masked_fill(c_word_mask, -1e30)
        pred_start = self.start_linear(emb_st).squeeze()
        pred_end = self.end_linear(emb_en).squeeze()
        return pred_start, pred_end
    
    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"


class QANetV2(nn.Module):
    def __init__(
        self, numChar, dim=128, dimChar=200, dimGlove=300, freeze=True
    ) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, freeze=freeze
        )

        self.embed_enc = EmbeddingEncoder(dimChar + dimGlove)
        self.context_query_attn = ContextQueryAttn(dim=dim)
        self.model_enc = ModelEncoderV2(embedDim=dim)
        # [B, sent_length, 401]
        self.start_linear = nn.Linear(dim, 1)
        self.end_linear = nn.Linear(dim, 1)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.embed_enc(self.input_emb(q))
        emb_c = self.embed_enc(self.input_emb(c))
        A, B = self.context_query_attn(emb_c, emb_q)
        out = self.model_enc(emb_c, A, B)
        pred_start = self.start_linear(out)
        pred_end = self.end_linear(out)
        return pred_start.squeeze(), pred_end.squeeze()

    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"


class EMA():

    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model, num_updates):
        decay = min(self.mu, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


if __name__ == "__main__":
    from torch.utils.data import DataLoader, Subset
    import torch.optim as optim
    from model import InputEmbedding
    from dataset import SQuADQANet
    from trainer import trainer, lr_scheduler_func

    datasetVersion = "v1"
    glove_dim = 300
    char_dim = 200
    glove_version = "6B"
    squadTrain = SQuADQANet("train", version=datasetVersion, glove_version=glove_version, glove_dim=glove_dim)
    # subsetTrain = squadTrain
    subsetTrain = Subset(squadTrain, [i for i in range(512)])
    # import pdb

    # pdb.set_trace()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model = QANet(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, freeze=True)

    # model = BaseClf2(numChar=squadTrain.charSetSize, dimChar=200, dimGlove=300)
    print(f"Model parameters: {model.count_params()}")
    model.to(device)

    trainLoader = DataLoader(subsetTrain, batch_size=32, shuffle=True)
    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.8, 0.999),
        eps=1e-7,
        lr=1e-3,
    )

    # exponential moving average
    ema = EMA(0.9999)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    
    warm_up_iters = 1000
    lr_func = lr_scheduler_func(warm_up_iters=warm_up_iters)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    criterion = nn.CrossEntropyLoss()

    # for epoch in range(2):
    #     for i, (contextDict, questionDict, label) in enumerate(trainLoader):
    #         print(epoch, i)
    #         model(contextDict, questionDict)
    #         quit()

    trainer(200, trainLoader, model, criterion, optimizer, lr_scheduler, device, ema)
