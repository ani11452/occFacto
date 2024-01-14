# FROM SPAGHETTI

from .occ_types import *
from abc import ABC 
import math
import torch.nn as nn
import torch.nn.functional as nnf


def torch_no_grad(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result

    return wrapper

class Model(nn.Module, ABC):

    def __init__(self):
        super(Model, self).__init__()
        self.save_model: Union[None, Callable[[nn.Module]]] = None

    def save(self, **kwargs):
        self.save_model(self, **kwargs)


class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class View(nn.Module):

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Dummy(nn.Module):

    def __init__(self, *args):
        super(Dummy, self).__init__()

    def forward(self, *args):
        return args[0]


class SineLayer(nn.Module):
    """
    From the siren repository
    https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.output_channels = out_features
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class MLP(nn.Module):

    def forward(self, x, *_):
        return self.net(x)

    def __init__(self, ch: Union[List[int], Tuple[int, ...]], act: nn.Module = nn.ReLU,
                 weight_norm=False):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(ch) - 1):
            layers.append(nn.Linear(ch[i], ch[i + 1]))
            if weight_norm:
                layers[-1] = nn.utils.weight_norm(layers[-1])
            if i < len(ch) - 2:
                layers.append(act(True))
        self.net = nn.Sequential(*layers)


class GMAttend(nn.Module):

    def __init__(self, hidden_dim: int):
        super(GMAttend, self).__init__()
        self.key_dim = hidden_dim // 8
        self.query_w = nn.Linear(hidden_dim, self.key_dim)
        self.key_w = nn.Linear(hidden_dim, self.key_dim)
        self.value_w = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=3)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = 1 / torch.sqrt(torch.tensor(self.key_dim, dtype=torch.float32))

    def forward(self, x):
        queries = self.query_w(x)
        keys = self.key_w(x)
        vals = self.value_w(x)
        attention = self.softmax(torch.einsum('bgqf,bgkf->bgqk', queries, keys))
        out = torch.einsum('bgvf,bgqv->bgqf', vals, attention)
        out = self.gamma * out + x
        return out


def recursive_to(item, device):
    if type(item) is T:
        return item.to(device)
    elif type(item) is tuple or type(item) is list:
        return [recursive_to(item[i], device) for i in range(len(item))]
    return item


# ____________________________________________________________________
# ____________________________ Attention _____________________________
# ____________________________________________________________________


class FeedForward(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward_interpolation(self, queries: T, keys: T, values: T, alpha: T, mask: TN = None) -> TNS:
        attention = torch.einsum('nhd,bmhd->bnmh', queries[0], keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        attention = attention * alpha[:, None, None, None]
        out = torch.einsum('bnmh,bmhd->nhd', attention, values).reshape(1, attention.shape[1], -1)
        return out, attention

    def forward(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        y = y if y is not None else x
        b_a, n, c = x.shape
        b, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b_a, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        if alpha is not None:
            out, attention = self.forward_interpolation(queries, keys, values, alpha, mask)
        else:
            attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
            attention = attention.softmax(dim=2)
            out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        x_, attention = self.attn(self.norm1(x), y, mask, alpha)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        x = x + self.attn(self.norm1(x), y, mask, alpha)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = FeedForward(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class DummyTransformer:

    @staticmethod
    def forward_with_attention(x, *_, **__):
        return x, []

    @staticmethod
    def forward(x, *_, **__):
        return x


class Transformer(nn.Module):

    def forward_with_attention(self, x, y: Optional[T] = None, mask: Optional[T] = None, alpha: TN = None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask, alpha)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y: TN = None, mask: TN = None, alpha: TN = None):
        for layer in self.layers:
            x = layer(x, y, mask, alpha)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.layers = nn.ModuleList([TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act,
                                                      norm_layer=norm_layer) for _ in range(num_layers)])
