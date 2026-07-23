from pathlib import Path

import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent      # outputs land next to this script
DATA = HERE.parent / "names.txt"            # dataset lives in the project root

words = open(DATA, "r").read().splitlines()
N = torch.zeros((27,27), dtype = torch.int32)
chars = sorted(list(set("".join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs,chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1][ix2] +=1

print(f"words in corpus      : {len(words)}")
print(f"vocabulary size      : {len(itos)} characters (26 letters + '.' as start/end token)")
print(f"bigrams counted      : {int(N.sum())}")
print(f"count matrix         : {tuple(N.shape)} (rows = first char, cols = second char)")
print(f"unseen bigrams       : {int((N == 0).sum())} of {N.numel()} cells are zero")

# the 5 most frequent bigrams, straight off the count matrix
top = torch.topk(N.flatten(), 5)
print("\nmost frequent bigrams:")
for count, flat_ix in zip(top.values.tolist(), top.indices.tolist()):
    i, j = divmod(flat_ix, 27)
    print(f"  {itos[i]}{itos[j]} : {count}")

plt.figure(figsize = (16,16))
plt.imshow(N, cmap = "Blues")
for i in range(27):
  for j in range(27):
    chstr = itos[i] + itos[j]
    plt.text(j,i,chstr, ha = "center", va = "bottom", color = "gray")
    plt.text(j,i,str(N[i,j].item()), ha = "center", va = "top", color = "gray")
plt.axis("off")
plt.savefig(HERE / "bigram_counts.svg", format = "svg", bbox_inches = "tight")
plt.close()
print("\nwrote bigram_counts.svg    (27x27 heatmap, every cell labeled)")

# P = [[]*27 for _ in range(27)]
# for ix in range(1):
#   p = N[ix].float()
#   P[ix] = p / p.sum()
#   print(p.sum())

P = (N+1).float()          # +1 is Laplace smoothing: no bigram gets probability 0
P = P/P.sum(1, keepdim = True)

# average negative log likelihood over every bigram in the corpus, weighted by count
nll = -(N * P.log()).sum() / N.sum()
print("\nmodel quality:")
print(f"  average negative log likelihood : {nll.item():.4f}")
print(f"  (a uniform 1-in-27 guesser scores {torch.tensor(27.0).log().item():.4f}; lower is better)")

g = torch.Generator().manual_seed(2147483647)

num_samples = 7
samples = []
for i in range(num_samples):
  out = []
  ix = 0
  while True:
    p = P[ix]
    ix  = int(torch.multinomial(p, num_samples = 1, replacement = True, generator = g).item())
    out.append(itos[ix])
    if ix == 0:
      break
  samples.append("".join(out))

print(f"\n{num_samples} names sampled from the bigram model:")
for name in samples:
  print(f"  {name}")

with open(HERE / "bigram_samples.txt", "w") as f:
  f.write(f"# makemore bigram (count-based) - {num_samples} sampled names\n")
  f.write(f"# words={len(words)} bigrams={int(N.sum())} smoothing=+1 seed=2147483647\n")
  f.write(f"# average negative log likelihood: {nll.item():.4f}\n")
  for name in samples:
    f.write(name + "\n")
print("\nwrote bigram_samples.txt")
