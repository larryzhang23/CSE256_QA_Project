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
    def __init__(self, numChar, dimChar=200, dimGlove=300, gloveVersion="6B", freeze=True, dropout=0.0):
        super().__init__()
        self.charEmbed = nn.Embedding(numChar, dimChar)
        glove = GloVe(name=gloveVersion, dim=dimGlove)
        self.wordPadIdx = glove.stoi["pad"]
        self.charPadIdx = numChar - 1
        self.gloveEmbed = nn.Embedding.from_pretrained(glove.vectors, freeze=freeze)
        self.conv = nn.Conv2d(dimChar, dimChar, (1, 5))
        self.hn = HighwayNetwork(dimChar + dimGlove)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # wordIdxTensor shape: [B, sent_length], charIdxTensor shape: [B, sent_length, 16]
        wordIdxTensor, charIdxTensor = x["wordIdx"], x["charIdx"]
        # charEmbedding shape: [B, sent_length, 16, char_dim]
        charEmbed = self.charEmbed(charIdxTensor)
        charEmbed = self.dropout1(charEmbed)
        charEmbed = F.relu(self.conv(charEmbed.permute(0, 3, 1, 2)))
        charEmbed, _ = torch.max(charEmbed, dim=-1)
        charEmbed = charEmbed.permute(0, 2, 1)  # new shape: [B,sent_length, char_ndim]
        # wordEmbedding shape: [B, sent_length, glove_dim]
        wordEmbed = self.gloveEmbed(wordIdxTensor)
        wordEmbed = self.dropout1(wordEmbed)
        # import pdb; pdb.set_trace()
        catEmbed = torch.cat(
            (wordEmbed, charEmbed), dim=2
        )  # [B, sent_length, glove_dim + char_dim]
        catEmbed = self.hn(catEmbed)
        catEmbed = self.dropout2(catEmbed)
        return catEmbed


class DepthWiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size=7, use_pad=True, dropout=0.0):
        """
        args:
            dim(int): 128
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
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        # self.dropout = nn.Dropout(p=0.1)
        self.layernorm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x shape: [B, sen_length, dim]
        x_res = x.permute(0, 2, 1)
        x = self.layernorm(x).permute(0, 2, 1)
        x = self.dropout(F.relu(self.pointwise(F.relu(self.depth(x)))))
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
        numConvLayers=4,
        nHeads=8,
        ker_size=7,
        dropout=0.0
    ):
        super().__init__()
        # convolution part
        conv = []
        for _ in range(numConvLayers):
            conv.append(DepthWiseConv1d(embedDim, kernel_size=ker_size, dropout=dropout))
        self.conv = nn.Sequential(*conv)

        # transformer part
        self.transformerBlock = nn.TransformerEncoderLayer(
            embedDim,
            nhead=nHeads,
            dim_feedforward=embedDim * 4,
            norm_first=True,
            batch_first=True,
            dropout=dropout,
        )
        self.nHeads = nHeads

    def forward(self, x, mask_idx=None):
        x = self.conv(x)
        # import pdb; pdb.set_trace()
        x = self.transformerBlock(x, src_key_padding_mask=mask_idx)
        return x


class ContextQueryAttn(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.w0 = nn.Linear(in_features=dim * 3, out_features=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        S = self.dropout(S)
        SS = self.dropout(SS)
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
        dropout=0.0,
    ):
        super().__init__()
        self.embedDim = embedDim
        blocks = []
        for _ in range(nBlocks):
            blocks.append(
                EmbeddingEncoder(
                    embedDim=embedDim,
                    numConvLayers=numConvLayers,
                    nHeads=nHeads,
                    ker_size=5,
                    dropout=dropout
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


class InputEmbedClf(nn.Module):
    def __init__(self, numChar, dimChar=20, dimGlove=50, version="v1", gloveVersion="6B") -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, gloveVersion=gloveVersion)
        # [B, sent_length, 400]
        if version == "v1":
            output_dim = 400
        else:
            output_dim = 401
        self.start_linear = nn.Linear(2 * (dimChar + dimGlove), output_dim)
        self.end_linear = nn.Linear(2 * (dimChar + dimGlove), output_dim)

    def forward(self, q, c):
        # [B, glove_dim + char_dim]
        emb_q = self.input_emb(q)
        emb_q = torch.mean(emb_q, dim=1)
        emb_c = self.input_emb(c)
        emb_c = torch.mean(emb_c, dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)
    
    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"


class EmbedEncClf(nn.Module):
    def __init__(self, numChar, dimChar=20, dimGlove=50, dim=128, with_mask=False, version="v1", gloveVersion="6B") -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, gloveVersion=gloveVersion
        )
        self.map = nn.Conv1d(dimChar + dimGlove, dim, kernel_size=1, bias=False)
        self.embed_enc = EmbeddingEncoder(dim)
        # [B, sent_length, 400]
        if version == "v1":
            output_dim = 400
        else:
            output_dim = 401
        self.start_linear = nn.Linear(2 * dim, output_dim)
        self.end_linear = nn.Linear(2 * dim, output_dim)
        self.with_mask = with_mask

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.map(self.input_emb(q).permute(0, 2, 1)).permute(0, 2, 1)
        emb_c = self.map(self.input_emb(c).permute(0, 2, 1)).permute(0, 2, 1)
        if self.with_mask:
            c_word_idx = c["wordIdx"]
            q_word_idx = q["wordIdx"]
            word_pad_idx = self.input_emb.wordPadIdx
            c_word_mask = c_word_idx == word_pad_idx 
            q_word_mask = q_word_idx == word_pad_idx
        else:
            c_word_mask = None
            q_word_mask = None 
        emb_q = self.embed_enc(emb_q, q_word_mask)
        emb_c = self.embed_enc(emb_c, c_word_mask)
        emb_q = torch.mean(emb_q, dim=1)
        emb_c = torch.mean(emb_c, dim=1)
        emb = torch.cat((emb_q, emb_c), dim=-1)
        return self.start_linear(emb), self.end_linear(emb)

    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"

class CQClf(nn.Module):
    def __init__(self, numChar, dimChar=20, dimGlove=50, dim=128, with_mask=False, version="v1", gloveVersion="6B") -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, gloveVersion=gloveVersion
        )
        self.map = nn.Conv1d(dimChar + dimGlove, dim, kernel_size=1, bias=False)
        self.embed_enc = EmbeddingEncoder(dim)
        # [B, sent_length, 128]
        self.context_query_attn = ContextQueryAttn(dim=dim)
        self.linear = nn.Linear(2 * dim, dim, bias=False)
        self.start_linear = nn.Linear(dim, 1)
        self.end_linear = nn.Linear(dim, 1)
        self.with_mask = with_mask

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.map(self.input_emb(q).permute(0, 2, 1)).permute(0, 2, 1)
        emb_c = self.map(self.input_emb(c).permute(0, 2, 1)).permute(0, 2, 1)
        if self.with_mask:
            c_word_idx = c["wordIdx"]
            q_word_idx = q["wordIdx"]
            word_pad_idx = self.input_emb.wordPadIdx
            c_word_mask = c_word_idx == word_pad_idx 
            q_word_mask = q_word_idx == word_pad_idx 
        else:
            c_word_mask = None
            q_word_mask = None
        emb_c = self.embed_enc(emb_c, c_word_mask)
        emb_q = self.embed_enc(emb_q, q_word_mask)
        emb_A, emb_B = self.context_query_attn(emb_c, emb_q, c_word_mask, q_word_mask)
        emb = torch.cat((emb_A, emb_B), dim=-1)
        emb = self.linear(emb)
        return self.start_linear(emb).squeeze(), self.end_linear(emb).squeeze()

    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"


class MACQClf(nn.Module):
    def __init__(self, numChar, dimChar=20, dimGlove=50, dim=128, with_mask=False, gloveVersion="6B", dropout=0.0) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, gloveVersion=gloveVersion, dropout=dropout
        )
        self.map = nn.Conv1d(dimChar + dimGlove, dim, kernel_size=1, bias=False)
        self.embed_enc = EmbeddingEncoder(dim)
        # [B, sent_length, 128]
        self.lnorm1 = nn.LayerNorm(dim)
        self.lnorm2 = nn.LayerNorm(dim)
        self.att = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True, dropout=dropout)
        self.lnorm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(in_features=dim, out_features=4*dim), nn.GELU(), nn.Linear(in_features=4*dim, out_features=dim))
        self.dropout_ffn = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.start_linear = nn.Conv1d(dim, 1, kernel_size=1, bias=False)
        self.end_linear = nn.Conv1d(dim, 1, kernel_size=1, bias=False)
        self.with_mask = with_mask

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.map(self.input_emb(q).permute(0, 2, 1)).permute(0, 2, 1)
        emb_c = self.map(self.input_emb(c).permute(0, 2, 1)).permute(0, 2, 1)
        if self.with_mask:
            c_word_idx = c["wordIdx"]
            q_word_idx = q["wordIdx"]
            word_pad_idx = self.input_emb.wordPadIdx
            c_word_mask = c_word_idx == word_pad_idx 
            q_word_mask = q_word_idx == word_pad_idx 
        else:
            c_word_mask = None
            q_word_mask = None
        emb_c = self.embed_enc(emb_c, c_word_mask)
        emb_q = self.embed_enc(emb_q, q_word_mask)
        emb_c = self.lnorm1(emb_c)
        emb_q = self.lnorm2(emb_q)
        emb = self.att(emb_c, emb_q, emb_q, key_padding_mask=q_word_mask)[0]
        emb = emb + self.dropout_ffn(self.ffn(self.lnorm_ffn(emb)))
        # emb_B = self.att(emb_c, emb_q, emb_q)[0]
        # emb = self.linear(emb)
        return self.start_linear(emb.permute(0, 2, 1)).squeeze(1), self.end_linear(emb.permute(0, 2, 1)).squeeze(1)

    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"
    

class TFCQClf(nn.Module):
    def __init__(self, numChar, dimChar=20, dimGlove=50, dim=128, with_mask=False, contextMaxLen=400, questionMaxLen=40, version="v1", gloveVersion="6B") -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, gloveVersion=gloveVersion
        )
        self.map = nn.Conv1d(dimChar + dimGlove, dim, kernel_size=1, bias=False)
        self.embed_enc = EmbeddingEncoder(dim)
        # [B, sent_length, 128]
        self.tf_layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=4*dim, batch_first=True, norm_first=True)
        if version == "v1":
            output_dim = 400
        else:
            output_dim = 401
        self.conv = nn.Conv1d(contextMaxLen + questionMaxLen, 1, kernel_size=1, bias=False)
        self.start_linear = nn.Linear(dim, output_dim)
        self.end_linear = nn.Linear(dim, output_dim)
        self.with_mask = with_mask

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        emb_q = self.map(self.input_emb(q).permute(0, 2, 1)).permute(0, 2, 1)
        emb_c = self.map(self.input_emb(c).permute(0, 2, 1)).permute(0, 2, 1)
        if self.with_mask:
            c_word_idx = c["wordIdx"]
            q_word_idx = q["wordIdx"]
            word_pad_idx = self.input_emb.wordPadIdx
            c_word_mask = c_word_idx == word_pad_idx 
            q_word_mask = q_word_idx == word_pad_idx 
            tf_word_mask = torch.cat((c_word_mask, q_word_mask), dim=1)
        else:
            c_word_mask = None
            q_word_mask = None
            tf_word_mask = None
        emb_c = self.embed_enc(emb_c, c_word_mask)
        emb_q = self.embed_enc(emb_q, q_word_mask)
        emb = torch.cat((emb_c, emb_q), dim=1)
        emb = self.tf_layer(emb, src_key_padding_mask=tf_word_mask)
        emb = self.conv(emb).squeeze(1)
        
        return self.start_linear(emb).squeeze(), self.end_linear(emb).squeeze()

    def count_params(self):
        params = filter(lambda x: x.requires_grad, self.parameters())
        num_params = sum(param.numel() for param in params)
        return f"Trainable Params: {(num_params / 1e6):.2f}M"
    
class QANet(nn.Module):
    def __init__(
        self, numChar, dim=128, dimChar=200, dimGlove=300, freeze=True, gloveVersion="6B", dropout=0.0, with_mask=True
    ) -> None:
        super().__init__()
        self.with_mask = with_mask
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, freeze=freeze, gloveVersion=gloveVersion, dropout=dropout
        )
        self.map = nn.Conv1d(dimChar + dimGlove, dim, kernel_size=1, bias=False)
        self.embed_enc = EmbeddingEncoder(embedDim=dim, dropout=dropout)
        self.context_query_attn = ContextQueryAttn(dim=dim, dropout=dropout)
        self.model_enc = ModelEncoder(embedDim=dim, dropout=dropout)
        # [B, sent_length, 401]
        self.start_linear = nn.Linear(2 * dim, 1)
        self.end_linear = nn.Linear(2 * dim, 1)

    def forward(self, c, q):
        # [B, glove_dim + char_dim]
        if self.with_mask:
            c_word_idx = c["wordIdx"]
            q_word_idx = q["wordIdx"]
            word_pad_idx = self.input_emb.wordPadIdx
            c_word_mask = c_word_idx == word_pad_idx 
            q_word_mask = q_word_idx == word_pad_idx 
        else:
            c_word_mask = None
            q_word_mask = None
        emb_q = self.embed_enc(self.map(self.input_emb(q).permute(0, 2, 1)).permute(0, 2, 1), q_word_mask)
        emb_c = self.embed_enc(self.map(self.input_emb(c).permute(0, 2, 1)).permute(0, 2, 1), c_word_mask)
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
        self, numChar, dim=128, dimChar=200, dimGlove=300, freeze=True, gloveVersion="6B"
    ) -> None:
        super().__init__()
        # [B, sent_length, glove_dim + char_dim]
        self.input_emb = InputEmbedding(
            numChar=numChar, dimChar=dimChar, dimGlove=dimGlove, freeze=freeze, gloveVersion=gloveVersion
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

# reference: https://github.com/BangLiu/QANet-PyTorch/blob/0ce11ca1494c6c30d61c0bf2b78907fe27369962/model/modules/ema.py
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