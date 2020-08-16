import torch
import torch.nn.functional as F
import torch.nn as nn

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false
    In place operation
    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
    matrices[:, indices[0], indices[1]] = maskval

class SelfAttention(nn.Module):
    def __init__(self,k,head=8, mask=False):
        super().__init__()
        self.k = k
        self.head = head
        self.mask = mask

        self.keys = nn.Linear(k,k*head, bias=False)
        self.queries = nn.Linear(k,k*head, bias=False)
        self.values = nn.Linear(k,k*head, bias=False)

        self.outhead = nn.Linear(k*head,k,bias=False)
    
    def forward(self,x):
        b,t,k = x.size()
        h=self.head
        queries = self.queries(x).view(b,t,h,k)
        keys = self.keys(x).view(b,t,h,k)
        values = self.values(x).view(b,t,h,k)

        queries = queries.transpose(1,2).contiguous().view(b*h,t,k)
        keys = keys.transpose(1,2).contiguous().view(b*h,t,k)
        values = values.transpose(1,2).contiguous().view(b*h,t,k)

        queries = queries/ (k**(1/4))
        keys = keys / (k**(1/4))

        dot = torch.bmm(queries,keys.transpose(1,2))

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim= 2)
        out = torch.bmm(dot,values).view(b,t,h,k)

        out= out.transpose(1,2).contiguous().view(b,t,h*k)
        return self.outhead(out)

class TransformerBlock(nn.Module):
    def __init__(self,k,head,mask_):
        super().__init__()

        self.attention = SelfAttention(k, head=head)

        self.mask = mask

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(nn.Linear(k,4*k), nn.ReLU(), nn.Linear(4*k,k))
    
    def forward(self,x):
        attend = self.attention(x)
        x = self.norm1(attend+x)

        mlp = self.ff(x)
        out = self.norm2(mlp+x)
        

        


    

