# from multiprocessing import Pool
import regex as re

def bpe(corpus, n) -> list[bytes,...]:
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

  word_freq, tokens = {}, [chr(i).encode('utf-8') for i in range(256)]
  tokens.append("<|endoftext|>")

  for w in re.findall(PAT, corpus):
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

    tokens.append(best)
    return {update(k):v for k,v in word_freq.items()}

  for _ in range(n): word_freq = merge()
  return {i:tk for i,tk in enumerate(tokens)}


def main():
  corpus = open("infinite-jest.txt", "r").read()

  vocab = bpe(corpus, 500)
  print(vocab)

if __name__ == "__main__":
  main()

