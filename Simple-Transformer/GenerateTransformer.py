import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import TransformerBlock, mask_

class GenerateTransformer(nn.Module):
    def __init__(self, k, head, depth, seq_length, num_tokens,dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        
        self.max_pool = max_pool

        self.token_emb = nn.Embedding(embedding_dim=k, num_embeddings = num_tokens)
        self.pos_emb = nn.Embedding(embedding_dim=k, num_embeddings = seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, head=head, mask = True))
        self.tblocks = nn.Sequential(*tblocks)

		# Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)
        self.do = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param x: A (batch, sequence length) integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        tokens = self.token_emb(x)
        b, t, e = tokens.size()

        positions = self.pos_emb(torch.arange(t, device="cuda"))[None, :, :].expand(b, t, e)
        x = tokens + positions

        x = self.tblocks(x)

        x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)

        return F.log_softmax(x, dim=2)
