from array import array
import os
from collections import Counter, defaultdict
import heapq
from pathlib import Path
import pickle

import regex as re


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BASE_VOCAB_SIZE = 257
Pair = tuple[bytes, bytes]
Model = tuple[dict[bytes, int], dict[Pair, int]]


def adjacent_pairs(word: tuple[bytes, ...]) -> Counter[Pair]:
  return Counter(zip(word, word[1:]))


def merge_pair(word: tuple[bytes, ...], pair: Pair) -> tuple[bytes, ...]:
  """Replace non-overlapping instances of pair from left to right."""
  merged = pair[0] + pair[1]
  out = []
  i = 0
  while i < len(word):
    if i + 1 < len(word) and (word[i], word[i+1]) == pair:
      out.append(merged)
      i += 2
    else:
      out.append(word[i])
      i += 1
  return tuple(out)


def bpe(corpus: str, ntoks: int) -> Model:
  if ntoks < BASE_VOCAB_SIZE:
    raise ValueError(f"ntoks must be at least {BASE_VOCAB_SIZE}")

  pretoken_freq = Counter(re.findall(PAT, corpus))
  words = [tuple(bytes([b]) for b in text.encode("utf-8")) for text in pretoken_freq]
  freqs = list(pretoken_freq.values())

  pair_freq: Counter[Pair] = Counter()
  pair_to_words: defaultdict[Pair, set[int]] = defaultdict(set)
  for word_id, (word, freq) in enumerate(zip(words, freqs)):
    for pair, occurrences in adjacent_pairs(word).items():
      pair_freq[pair] += occurrences * freq
      pair_to_words[pair].add(word_id)
  pair_heap = [(freq, pair) for pair, freq in pair_freq.items()]
  heapq.heapify_max(pair_heap)

  vocab = {bytes([i]): i for i in range(256)}
  vocab[b"<|endoftext|>"] = 256
  merge_ranks: dict[Pair, int] = {}

  for _ in range(ntoks - BASE_VOCAB_SIZE):
    if not pair_freq:
      break

    while pair_heap and pair_freq.get(pair_heap[0][1]) != pair_heap[0][0]:
      heapq.heappop_max(pair_heap)
    _, best = heapq.heappop_max(pair_heap)
    affected = tuple(pair_to_words.pop(best))
    changed_pairs = set()

    for word_id in affected:
      old_word = words[word_id]
      freq = freqs[word_id]
      old_pairs = adjacent_pairs(old_word)

      for pair, occurrences in old_pairs.items():
        changed_pairs.add(pair)
        remaining = pair_freq[pair] - occurrences * freq
        if remaining:
          pair_freq[pair] = remaining
        else:
          pair_freq.pop(pair)

        word_ids = pair_to_words.get(pair)
        if word_ids is not None:
          word_ids.discard(word_id)
          if not word_ids:
            pair_to_words.pop(pair, None)

      new_word = merge_pair(old_word, best)
      words[word_id] = new_word
      for pair, occurrences in adjacent_pairs(new_word).items():
        changed_pairs.add(pair)
        pair_freq[pair] += occurrences * freq
        pair_to_words[pair].add(word_id)

    for pair in changed_pairs:
      if pair in pair_freq:
        heapq.heappush_max(pair_heap, (pair_freq[pair], pair))

    merge_ranks[best] = len(merge_ranks)
    token = best[0] + best[1]
    if token not in vocab:
      vocab[token] = len(vocab)

  return vocab, merge_ranks


def encode(text: str, model: Model) -> list[int]:
  vocab, merge_ranks = model
  encoded = []

  for pretoken in re.findall(PAT, text):
    word = tuple(bytes([b]) for b in pretoken.encode("utf-8"))

    while len(word) > 1:
      ranked_pairs = (
        (merge_ranks[pair], pair)
        for pair in zip(word, word[1:])
        if pair in merge_ranks
      )
      try:
        _, best = min(ranked_pairs)
      except ValueError:
        break
      word = merge_pair(word, best)

    encoded.extend(vocab[token] for token in word)

  return encoded

def decode(fname, vocab):
  inv_vocab = {v:k for k,v in vocab.items()}
  arr = array("H")
  arr.fromfile(open(fname, "rb"), os.path.getsize(fname)//2)
  return b"".join(inv_vocab[x] for x in arr).decode('utf-8')

def main():
  corpus = Path("infinite-jest.txt").read_text()
  model_path = Path("vocab.pkl")
  model = None

  if model_path.is_file():
    cached = pickle.loads(model_path.read_bytes())
    if isinstance(cached, tuple) and len(cached) == 2:
      model = cached

  if model is None:
    model = bpe(corpus, 20000)
    model_path.write_bytes(pickle.dumps(model))

  vocab, _ = model
  print(len(vocab))

  source = Path("test.txt").read_text()
  ids = array("H", encode(source, model))
  ids.tofile(open("tokens.bin", "wb"))

  decoded = decode("tokens.bin", vocab)

if __name__ == "__main__":
  main()

