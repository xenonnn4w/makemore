"""makemore part 3: activations, gradients and what a deep net does to them.

Same task as the MLP in ../mlp, but six layers deep instead of one. At this
depth the init scaling stops being a detail: get it wrong and the tanh layers
saturate or the gradients vanish. tools.py is how we see that happening.
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F

import tools
from layers import Linear, Tanh

HERE = Path(__file__).resolve().parent      # outputs land next to this script
DATA = HERE.parent / "names.txt"            # dataset lives in the project root

# Flip to False for a real training run. True keeps the debug behaviour: a
# short run that retains activation gradients so the diagnostics have data.
DEBUG = True

words = open(DATA, "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)

block_size = 3
def build_dataset(words):
  X, Y = [], []
  # ...arman .
  for w in words:
    # print(w)
    context = [0]*block_size
    for ch in w+'.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      # print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:]+[ix]

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  return X,Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1]) # training set 80 %
Xdev, Ydev = build_dataset(words[n1:n2]) # dev set 10 %
Xte, Yte = build_dataset(words[n2:]) # testing set 10 %

print(f"words in corpus      : {len(words)}")
print(f"vocabulary size      : {vocab_size} characters (26 letters + '.')")
print(f"context block size   : {block_size} characters")
print(f"train examples       : {tuple(Xtr.shape)} -> {tuple(Ytr.shape)}")
print(f"dev examples         : {tuple(Xdev.shape)} -> {tuple(Ydev.shape)}")
print(f"test examples        : {tuple(Xte.shape)} -> {tuple(Yte.shape)}")

n_embed = 10 # the dimensionality of the character embedding vectors
n_hidden = 100 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size,n_embed), generator=g)
# we can remove this Tanh(), but that will leads to a calculated behaviour i.e. linear i/p and o/p; the the activation function help it somewhat so we can teach a
# nerual network to work on arbitary data, given we trained it. idk its jittey but for now bear this
layers = [
    Linear(n_embed*block_size,n_hidden, generator=g), Tanh(),
    Linear(n_hidden,n_hidden, generator=g), Tanh(),
    Linear(n_hidden,n_hidden, generator=g), Tanh(),
    Linear(n_hidden,n_hidden, generator=g), Tanh(),
    Linear(n_hidden,n_hidden, generator=g), Tanh(),
    Linear(n_hidden, vocab_size, generator=g),
]


"""
this above MLP have following

Input(30)->Layer1(100)->Tanh()->Layer2(100)->Tanh()->Layer3(100)->Tanh()->Layer4(100)->Tanh()->Layer5(100)->Tanh()->Layer6(27)

Input is 30 coz n_embed = 10 and block_size = 3 i.e. [e,m,m] -> [0.1,0.4,45,.....5.6] this single example will have 30 data point

now let say output is [e,m,m] -> [a] i.e. we can have 27 possible characters thus output as 27 (we have 0 -> '.')

"""


with torch.no_grad():
  # last layer: make less confident coz we dont want that it randomly assign high confidence an character which is not supposed to be (i.e. learning will have to make a lot of effort to correct itself)
  # conclusion: we can have smooth start in training
  last_layer = layers[-1]
  assert isinstance(last_layer, Linear)
  last_layer.weight *= 0.1

  # all other layers: apply gain (as Tanh() try to squash them overtime, i.e. std will come near to 0, so mathmatically calculated gain is multiplied)
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *=  5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
for p in parameters:
  p.requires_grad = True

print(f"\nembedding dim        : {n_embed}")
print(f"hidden units         : {n_hidden} x {sum(isinstance(l, Tanh) for l in layers)} tanh layers")
print(f"trainable parameters : {sum(p.nelement() for p in parameters)}")
print(f"init                 : 1/sqrt(fan_in), x5/3 tanh gain, last layer x0.1")


# same optimization as last time
max_steps = 1000 if DEBUG else 200000
batch_size = 32
lossi = []
ud = []

print(f"\ntraining for {max_steps} steps, batch size {batch_size}, lr 0.1 -> 0.01 at step 150000"
      + ("  [DEBUG: short run]" if DEBUG else ""))

for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  emb = C[Xb] # embed the characters into vectors
  x = emb.view(emb.shape[0], -1) # concatenate the vectors
  for layer in layers:
    x = layer(x)
  loss = F.cross_entropy(x, Yb) # loss function

  # backward pass
  if DEBUG:
    for layer in layers:
      layer.out.retain_grad() # keeps the activation gradients that tools.py plots
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    assert p.grad is not None
    p.data += -lr * p.grad

  # track stats
  if i % (100 if DEBUG else 10000) == 0: # print every once in a while
    print(f'  {i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

print(f'  {max_steps:7d}/{max_steps:7d}: {loss.item():.4f}  (final minibatch, noisy)')


# diagnostics: four plots that say whether this network is actually trainable
print()
report = tools.report_all(layers, parameters, ud, lossi)


@torch.no_grad()
def split_loss(X, Y):
  """Full-batch cross-entropy loss on one dataset split."""
  x = C[X].view(X.shape[0], -1)
  for layer in layers:
    x = layer(x)
  return F.cross_entropy(x, Y).item()


train_loss = split_loss(Xtr, Ytr)
dev_loss = split_loss(Xdev, Ydev)
test_loss = split_loss(Xte, Yte)
print("final losses (bigram counts scored 2.4546, the 1-layer MLP 2.3332):")
print(f"  train : {train_loss:.4f}")
print(f"  dev   : {dev_loss:.4f}")
print(f"  test  : {test_loss:.4f}")
if DEBUG:
  print("  note: DEBUG run, only 1000 steps - these are far from converged")


# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
num_samples = 20
samples = []
for _ in range(num_samples):
  out = []
  context = [0] * block_size # initialize with all ...
  while True:
    x = C[torch.tensor([context])].view(1, -1)
    for layer in layers:
      x = layer(x)
    probs = F.softmax(x, dim=1)
    ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
    context = context[1:] + [ix]
    out.append(ix)
    if ix == 0:
      break
  samples.append(''.join(itos[i] for i in out))

print(f"\n{num_samples} names sampled from the trained model:")
for name in samples:
  print(f"  {name}")

with open(HERE / "training_report.txt", "w") as f:
  f.write(f"# makemore part 3 - deep MLP diagnostics\n")
  f.write(f"# n_embed={n_embed} n_hidden={n_hidden} layers={len(layers)} "
          f"params={sum(p.nelement() for p in parameters)} steps={max_steps} debug={DEBUG}\n")
  f.write(f"# losses: train={train_loss:.4f} dev={dev_loss:.4f} test={test_loss:.4f}\n\n")
  f.write("\n".join(report))
  f.write(f"\n{num_samples} sampled names\n")
  for name in samples:
    f.write(f"  {name}\n")
print("\nwrote training_report.txt")
