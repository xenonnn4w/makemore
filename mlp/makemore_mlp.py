import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent      # outputs land next to this script
DATA = HERE.parent / "names.txt"            # dataset lives in the project root

words = open(DATA, "r").read().splitlines()

chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

# building the dataset

block_size = 3
X, Y = [], []
for w in words:
    # print(w)
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix]

X = torch.tensor(X)
Y = torch.tensor(Y)


def build_dataset(words):
    block_size = 3
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context), '--->', itos[ix])
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])  # training set
Xdev, Ydev = build_dataset(words[n1:n2])  # dev set
Xte, Yte = build_dataset(words[n2:])  # testing set

print(f"words in corpus      : {len(words)}")
print(f"vocabulary size      : {len(itos)} characters (26 letters + '.')")
print(f"context block size   : {block_size} characters")
print(f"train examples       : {tuple(Xtr.shape)} -> {tuple(Ytr.shape)}")
print(f"dev examples         : {tuple(Xdev.shape)} -> {tuple(Ydev.shape)}")
print(f"test examples        : {tuple(Xte.shape)} -> {tuple(Yte.shape)}")

# C = torch.randn((27, 2))
# emb = C[X]
# W1 = torch.randn((6,100))
# b1 = torch.rand(100)

# emb.view(32,6) @ W1 + b1 #  dimensioons mismatch
# h = torch.tanh(emb.view(32,6) @ W1 + b1)

# method 1
# torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]],1).shape

# method 2
# torch.cat(torch.unbind(emb,1),1).shape

# method 3 and best one ()
# emb.view(32,6).shape

lr = torch.linspace(0.001,1,1000) #learning rate
lre = torch.linspace(-3,0,1000) #learning rate exponent
lrs = 10**lre # actual learning rates

# Everyhting all together
g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2), generator = g)
W1 = torch.randn((6,300), generator = g)
b1 = torch.randn(300, generator = g)
W2 = torch.randn((300,27), generator = g)
b2 = torch.randn(27, generator = g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
  p.requires_grad = True

print(f"\nembedding dim        : {C.shape[1]}")
print(f"hidden units         : {W1.shape[1]}")
print(f"trainable parameters : {sum(p.nelement() for p in parameters)}")

lri = []
lossi = []
stepi = []
max_steps = 100000
batch_size = 32
print(f"\ntraining for {max_steps} steps, batch size {batch_size}, lr 0.1 -> 0.01 at step 10000")
for i in range(max_steps):

  # mini batch
  ix = torch.randint(0,Xtr.shape[0],(batch_size,))
  #forward pass
  emb  = C[Xtr[ix]] #(32,3,2)
  h = torch.tanh(emb.view(-1,6) @ W1 + b1) #(32,100)
  logits = h @ W2 + b2 #(32,27)
  # counts = logits.exp()
  # prob = counts/counts.sum(1, keepdims = True)
  # loss = - prob[torch.arange(32),Y].log().mean()
  # replacign above by cross_entropy as
  loss = F.cross_entropy(logits,Ytr[ix])

  # backward pass
  for p in parameters:
    p.grad = None

  loss.backward()

  # nudge the weights such that we are minimizing the loss or going in negative gradient direction hehe
  # lr = lrs[i]
  lr = 0.1 if i<10000 else 0.01 # we found form the graph the optimal learning rate
  for p in parameters:
    assert p.grad is not None # every parameter took part in the forward pass
    p.data += -1 * lr * p.grad

  stepi.append(i)
  # lri.append(lre[i])
  lossi.append(loss.item())

  if i % 10000 == 0:
    print(f"  step {i:6d}/{max_steps}  minibatch loss {loss.item():.4f}  lr {lr}")

print(f"  step {max_steps:6d}/{max_steps}  minibatch loss {loss.item():.4f}  (final minibatch, noisy)")


@torch.no_grad()
def split_loss(X, Y):
  """Full-batch cross-entropy loss on one dataset split (no gradient tracking)."""
  emb = C[X]                                # (N, block_size, embedding_dim)
  h = torch.tanh(emb.view(-1, 6) @ W1 + b1) # (N, hidden)
  logits = h @ W2 + b2                      # (N, vocab)
  return F.cross_entropy(logits, Y).item()


train_loss = split_loss(Xtr, Ytr)
dev_loss = split_loss(Xdev, Ydev)
test_loss = split_loss(Xte, Yte)
print("\nfinal losses (lower is better, ~2.45 is a decent MLP bigram-beater):")
print(f"  train : {train_loss:.4f}")
print(f"  dev   : {dev_loss:.4f}")
print(f"  test  : {test_loss:.4f}")
print(f"  train/dev gap {dev_loss - train_loss:+.4f} -> {'overfitting' if dev_loss - train_loss > 0.1 else 'still underfitting, model has room to grow'}")

# loss curve over training
plt.figure(figsize=(10,5))
plt.plot(stepi, lossi, linewidth=0.5)
plt.xlabel("step")
plt.ylabel("minibatch cross-entropy loss")
plt.title("training loss")
plt.savefig(HERE / "mlp_loss_curve.svg", format="svg", bbox_inches="tight")
plt.close()
print("\nwrote mlp_loss_curve.svg   (minibatch loss vs step)")

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid(True, which="minor")
plt.title("learned character embeddings (dims 0 and 1)")
plt.savefig(HERE / "mlp_embeddings.svg", format="svg", bbox_inches="tight")
plt.close()
print("wrote mlp_embeddings.svg   (vowels should cluster together)")

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

num_samples = 20
samples = []
for _ in range(num_samples):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = int(torch.multinomial(probs, num_samples=1, generator=g).item())
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break

    samples.append(''.join(itos[i] for i in out))

print(f"\n{num_samples} names sampled from the trained model:")
for name in samples:
    print(f"  {name}")

with open(HERE / "mlp_samples.txt", "w") as f:
    f.write(f"# makemore MLP - {num_samples} sampled names\n")
    f.write(f"# block_size={block_size} embedding_dim={C.shape[1]} hidden={W1.shape[1]} "
            f"params={sum(p.nelement() for p in parameters)} steps={max_steps}\n")
    f.write(f"# losses: train={train_loss:.4f} dev={dev_loss:.4f} test={test_loss:.4f}\n")
    for name in samples:
        f.write(name + "\n")
print("\nwrote mlp_samples.txt")
