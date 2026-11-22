from tinygrad import Tensor, dtypes, nn
from array import array
import pickle
import numpy as np
from bpe import encode

# is actually linear cause no affine bias lol
class Linear:
  def __init__(self, in_features, out_features):
    self.W = Tensor.normal(out_features, in_features, std=2./(in_features + out_features))
  def __call__(self, x: Tensor): return x @ self.W.T

class Embedding: # (seq_len, vocab_size) => (seq_len, vocab_size, model_dim)
  def __init__(self, vocab_size, embed_dim):
    self.W = Tensor.normal(vocab_size, embed_dim)
  def __call__(self, x): return self.W[x]

# multi-head, causal masked
class Attention:
  def __init__(self, embed_dim, n_head, n_kv_head, head_dim):
    self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, head_dim
    self.proj_q = Linear(embed_dim, n_head*head_dim)
    self.proj_k = Linear(embed_dim, n_kv_head*head_dim)
    self.proj_v = Linear(embed_dim, n_kv_head*head_dim)
    self.W_out = Linear(n_kv_head*head_dim, embed_dim)

  def __call__(self, x: Tensor, cos_sin):
    B, L, dim = x.shape
    # convert to qkv heads?
    # when do I apply the rotary
    q = self.proj_q(x).reshape(B, L, self.n_head, self.head_dim)
    k = self.proj_k(x).reshape(B, L, self.n_kv_head, self.head_dim)
    v = self.proj_v(x).reshape(B, L, self.n_kv_head, self.head_dim)

    # rope should only be applied acress head dims, heads are part of batch..
    cos, sin = cos_sin # truncate to current seq len
    cos, sin = cos[:, :L], sin[:, :L]
    q, k = apply_rope(q, (cos,sin)), apply_rope(k, (cos,sin))

    q, k, v = [t.transpose(0, 2) for t in (q,k,v)]
    qk = (q @ k.transpose(-2,-1)) / np.sqrt(self.head_dim)
    # this broadcasts across heads? (bchsz, heads, seqln, n_kv_head*head_dim)
    mask = Tensor.full(qk.shape, True, dtypes.bool).tril()
    # todo: cache this, max context len
    qk = mask.where(qk, qk - np.inf) # causal masking
    a = (qk.softmax() @ v).transpose(0,2).flatten(2)
    return self.W_out(a)

class MLP:
  def __init__(self, embed_dim, d_hidden):
    self.W1, self.W2 = [Tensor.normal(d_hidden, embed_dim) for _ in range(2)]
    self.W3 = Tensor.normal(embed_dim, d_hidden)
  def __call__(self, x: Tensor):
    return ((x @ self.W1.T).swish() * (x @ self.W2.T)) @ self.W3.T

class TransformerBlock:
  def __init__(self, embed_dim, n_head, n_kv_head, head_dim, cos_sin):
    d_ff = int((8/3)*embed_dim)
    self.cos_sin = cos_sin
    self.attention = Attention(embed_dim, n_head, n_kv_head, head_dim)
    self.mlp = MLP(embed_dim, d_ff)
    self.attn_norm, self.mlp_norm = nn.RMSNorm(embed_dim), nn.RMSNorm(embed_dim)
  def __call__(self, x: Tensor):
    h = x + self.attention(self.attn_norm(x), self.cos_sin)
    return h + self.mlp(self.mlp_norm(h))

# TODO: compute for max ctx len and clip when applying
def precompute_rotary_embeddings(seq_len, head_dim, base=10000) -> tuple[Tensor,Tensor]:
  channels = Tensor.arange(head_dim, step=2)
  inv_freq = 1.0 / (base ** (channels / head_dim))
  t = Tensor.arange(seq_len)
  # use einsum instead of outer. what is an outer matrix product?
  # we want a pairwise rotation matrix for every time step and hcannel?
  freqs = Tensor.einsum('i,j->ij', t, inv_freq)
  cos, sin = freqs.cos(), freqs.sin()
  # helps w/ broadcasting?
  return cos[None, :, None, :], sin[None, :, None, :]

def apply_rope(x: Tensor, cos_sin):
  cos, sin = cos_sin
  d = x.shape[-1] // 2
  # split dimension into pairs to rotate 2d?
  x1, x2 = x[..., :d], x[..., d:]
  y1 = x1 * cos + x2 * sin
  y2 = x1 * (-sin) + x2 * cos
  return y1.cat(y2, dim=3)

class Transformer:
  def __init__(self, embed_dim, n_layers, n_heads, n_kv_heads, head_dim, max_ctx, vocab_size):
    self.embedding = Embedding(vocab_size, embed_dim)
    cos_sin = precompute_rotary_embeddings(max_ctx, head_dim)
    self.layers = [TransformerBlock(embed_dim, n_heads, n_kv_heads, head_dim, cos_sin) for _ in range(n_layers)]
  def __call__(self, x:Tensor):
    # TODO: manage/save kv cache?, measure perplexity...
    x = self.embedding(x)
    out = x.sequential(self.layers)
    return out

if __name__ == "__main__":
  vocab = pickle.load(open("vocab.pkl", "rb"))
  prompt = "Was that life? Well then! Once more!"
  x = Tensor(encode(prompt, vocab), dtype=dtypes.uint32).unsqueeze(0)

  model = Transformer(256, 4, 3, 3, 64, 128, len(vocab[1]))
  logits = model(x)
  print(logits.shape, logits.numpy())
