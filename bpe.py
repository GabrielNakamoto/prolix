from multiprocessing import Pool
from functools import partial
from pathlib import Path
import regex as re, pickle

def worker(words, nmerges):
  word_freq, merges = {}, []
  for w in words:
    b = tuple(c.encode('utf-8') for c in w)
    word_freq[b] = word_freq.setdefault(b,0) + 1

  last_updated, pair_freq = {}, {}
  def count_pairs(w):
    for a,b in zip(w, w[1:]):
      k = b"".join((a,b))
      pair_freq[k] = pair_freq.setdefault(k,0)+word_freq[w]

  for w in word_freq.keys(): count_pairs(w)

  def merge():
    for w,olds in last_updated.items():
      for p in olds: pair_freq[p] -= word_freq[w] # could be better
      count_pairs(w)
    last_updated.clear()

    best = max(pair_freq.items(), key=lambda x: (x[1], x[0]))[0]

    def update(w:tuple[bytes,...]):
      idxs = [i for i,(a,b) in enumerate(zip(w, w[1:])) if b"".join((a,b)) == best]
      old_pairs = tuple(b"".join((a,b)) for a,b in zip(w, w[1:]))
      for j,i in enumerate(idxs): w = w[:i-j] + (best,) + w[i-j+2:]
      if len(idxs): last_updated[w] = old_pairs
      return w

    merges.append(best)
    return {update(k):v for k,v in word_freq.items()}

  for _ in range(nmerges): word_freq = merge()
  return merges

def bpe(corpus, ntoks, nthreads) -> list[bytes,...]:
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  words = re.findall(PAT, corpus)
  nw = len(words) // nthreads
  batches = [words[i*nw:(i+1)*nw] if i < (nthreads-1) else words[i*nw:] for i in range(nthreads)]

  tokens = [bytes([i]) for i in range(256)]
  tokens.append(b"<|endoftext|>")

  with Pool(nthreads) as p:
    merges = []
    for ms in p.map(partial(worker, nmerges=(ntoks//nthreads)), batches):
      merges.extend(ms)
    tokens.extend(set(merges)) # dedup

  return {tk:i for i,tk in enumerate(tokens)}

def tokenize(text, vocab):
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  words = re.findall(PAT, text)

  def merge(bs):
    pair = b"".join(bs[0:2])
    if len(bs) == 1: return bs
    elif pair in vocab:
      return merge([pair] + bs[2:])
    else: return [bs[0]] + merge(bs[1:])

  for w in words:
    b = [bytes([b]) for b in w.encode('utf-8')]
    print(merge(b))
    out = [vocab[p] for p in merge(b)]
    print(w, out)

def main():
  corpus = open("infinite-jest.txt", "r").read()

  if Path("vocab.pkl").is_file():
    vocab = pickle.load(open("vocab.pkl", "rb"))
  else:
    vocab = bpe(corpus, 25000, 8)
    pickle.dump(vocab, open("vocab.pkl", "wb"))

  print(vocab)

  # try tokenizing, whats the compression ratio?
  source = open("test.txt", "r").read()
  tokenize(source, vocab)

if __name__ == "__main__":
  main()

