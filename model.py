from tinygrad import Tensor, dtypes, Context, nn
from array import array
import pickle, bpe
import numpy as np

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
  cos, sin = cos[None, :, None, :], sin[None, :, None, :]
  cos.is_param, sin.is_param = False, False
  return cos, sin

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
    self.norm = nn.LayerNorm(embed_dim)
    self.out_proj = Linear(embed_dim, embed_dim)
  def __call__(self, x:Tensor):
    # TODO: manage/save kv cache?, measure perplexity...
    x = self.embedding(x)
    out = x.sequential(self.layers)
    return self.out_proj(self.norm(out))

def cosine_annealing_schedule(t, lr_min, lr_max, t_warm, t_total):
  if t < to_warm: return t/t_warm * lr_max
  elif t > t_total: return lr_min
  else: return lr_min + 1/2 * (1 + np.cos((t-t_warm)/(t_total-t_warm)*np.pi)) * (lr_max - lr_min)

def load_dataset(filename, batch_size, context_len):
  vocab = pickle.load(open("vocab.pkl", "rb"))
  text = open(filename, "r").read()
  tokens = np.array(bpe.encode(text, vocab), dtype=np.int32)
  tokens = tokens[:(tokens.shape[0]//context_len)*context_len].reshape(-1, context_len)
  while True:
    idx = np.random.randint(0, tokens.shape[0]-context_len+1, batch_size)
    yield Tensor(tokens[idx][:, :-1], dtype=dtypes.uint32), Tensor(tokens[idx][:, 1:], dtype=dtypes.uint32)

if __name__ == "__main__":
  vocab = pickle.load(open("vocab.pkl", "rb"))
  model = Transformer(256, 4, 3, 3, 64, 512, len(vocab[1]))
  optim = nn.optim.AdamW(nn.state.get_parameters(model))
  data = load_dataset("infinite-jest.txt", 32, 256)

  # TODO: custom muon/adamw optimizer, gradient clipping
  @Context(TRAINING=1)
  def train_step():
    train_x, train_y = next(data)
    optim.zero_grad()
    loss = model(train_x).sparse_categorical_crossentropy(train_y).backward()
    optim.step()
    return loss.item()

  for epoch in range(20):
    # TODO: update lr
    loss = train_step()
    print(f"epoch={epoch}, loss={loss:.4f}")

