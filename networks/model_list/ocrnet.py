import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['OCRNET', 'ocrnet']

#transformer functions taken from here https://github.com/pbloem/former/
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

def contains_nan(tensor):
    return bool((tensor != tensor).sum() > 0)

class SelfAttention(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = dot / math.sqrt(e) # dot contains b*h  t-by-t matrices with raw self-attention logits

        assert dot.size() == (b*h, t, t), f'Matrix has size {dot.size()}, expected {(b*h, t, t)}.'

        if self.mask: # mask out the lower half of the dot matrix,including the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2) # dot now has row-wise self-attention probabilities

        assert not contains_nan(dot[:, 1:, :]) # only the forst row may contain nan

        if self.mask == 'first':
            dot = dot.clone()
            dot[:, :1, :] = 0.0
            # - The first row of the first attention matrix is entirely masked out, so the softmax operation results
            #   in a division by zero. We set this row to zero by hand to get rid of the NaNs

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, emb, heads, mask=None, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x

#the cnn net is taken from https://github.com/JackonYang/captcha-tensorflow
class OCRNET(nn.Module):

    def __init__(self, num_classes=None, nr_digits=8, transformer_depth=4):
        super(OCRNET, self).__init__()
        self.num_classes = num_classes
        self.nr_digits = nr_digits
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 3, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(3),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, self.nr_digits * self.num_classes),
            #nn.Softmax(dim=1) #cross entropy expects raw inputs in pytorch
        )

        self.transformers = []

        for i in range(transformer_depth):
            # num_classes because we reshape it in forward
            self.transformers.append(TransformerBlock(self.num_classes, 4))

        self.transformers = nn.Sequential(*self.transformers)

        # num_classes because we add it to transformers output
        self.pos_emb = nn.Sequential(
            nn.Embedding(self.nr_digits, self.num_classes)
        )

    def init_weights(self):
        def init_sequential(m):
            if type(m) in [nn.Conv2d, nn.Linear]:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0.01)

        self.features.apply(init_sequential)
        self.classifier.apply(init_sequential)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view((x.shape[0], self.nr_digits, self.num_classes))

        #generate positions for embeddings
        positions = torch.arange(self.nr_digits)
        positions = self.pos_emb(positions)[None, :, :].expand(x.shape[0], self.nr_digits, self.num_classes)
        x = x + positions
        #apply transformers
        x = self.transformers(x)
        return x

def ocrnet(pretrained=False, **kwargs):
    model = OCRNET(**kwargs)
    if pretrained:
        model_path = '../checkpoint_v1_1.pth.tar'
        if torch.cuda.is_available():
            pretrained_model = torch.load(model_path)
        else:
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_model['state_dict'])
    return model
