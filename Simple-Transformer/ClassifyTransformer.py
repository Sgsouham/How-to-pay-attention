import torch 
import torch.nn as nn
import torch.nn.functional as F

from utils import TransformerBlock

class ClassifyTransformer(nn.Module):
    def __init__(self, k, head, depth, mask, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, wide=False):
        super().__init__()
        self.seq_length = seq_length
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.max_pool = max_pool
        self.mask = mask

        self.token_emb = nn.Embedding(embedding_dim=k, num_embeddings = num_tokens)
        self.pos_emb = nn.Embedding(embedding_dim=k, num_embeddings = seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, head=head, mask=False))
        self.tblocks = nn.Sequential(*tblocks)

		# Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)
        self.do = nn.Dropout(dropout)
    
    def forward(self,x):

        tokens = self.token_emb(x)
        b, t, e = tokens.size()

        positions = self.pos_emb(torch.arange(t, device="cuda"))[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1) # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)

        

