from pathlib import Path

words = open(Path(__file__).resolve().parent.parent / "names.txt", "r").read().splitlines()
words[:8]

# Commented out IPython magic to ensure Python compatibility.
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

torch.set_num_threads(1)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

block_size = 8
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
  print(X.shape,Y.shape)
  return X,Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1]) # training set 80 %
Xdev, Ydev = build_dataset(words[n1:n2]) # dev set 10 %
Xte, Yte = build_dataset(words[n2:]) # testing set 10 %

# n_embed = 10
# n_hidden = 200
# # Everyhting all together
# g = torch.Generator().manual_seed(2147483647)
# C = torch.randn((vocab_size, n_embed), generator = g) # [27,10]
# W1 = torch.randn((n_embed*block_size, n_hidden), generator = g) * (5/3)/((n_embed*block_size)**0.5)  # [30,200]
# b1 = torch.randn(n_hidden, generator = g) *.01 # [200]
# W2 = torch.randn((n_hidden, vocab_size), generator = g) * 0.01 # [200,27]
# b2 = torch.randn(vocab_size, generator = g) * 0 # [27]

# bngain = torch.ones((1,n_hidden)) # [1,200]
# bnbias = torch.zeros((1,n_hidden)) # [1,200]
# bnmean_running = torch.zeros((1,n_hidden)) # [1,200]
# bnstd_running = torch.ones((1,n_hidden)) # [1,200]

# parameters = [C, W1, b1, W2, b2]
# for p in parameters:
#   p.requires_grad = True
# sum(p.nelement() for p in parameters)

# Let's train a deeper network
# The classes we create here are the same API as nn.Module in PyTorch
from layers import BatchNorm1d, Embedding, FlattenConsecutive, Linear, Sequential, Tanh

n_embed = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP
g = torch.Generator().manual_seed(2147483647)
# C = torch.randn((vocab_size,n_embed), generator=g)
# we can remove this Tanh(), but that will leads to a calculated behaviour i.e. linear i/p and o/p; the the activation function help it somewhat so we can teach a
# nerual network to work on arbitary data, given we trained it. idk its jittey but for now bear this
model = Sequential([
    Embedding(vocab_size, n_embed),
    FlattenConsecutive(2), Linear(n_embed*2,n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2,n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(n_hidden*2,n_hidden, bias = False), BatchNorm1d(n_hidden), Tanh(),
    Linear(n_hidden, vocab_size),
])



"""
this above MLP have following

Input(30)->Layer1(100)->Tanh()->Layer2(100)->Tanh()->Layer3(100)->Tanh()->Layer4(100)->Tanh()->Layer5(100)->Tanh()->Layer6(27)

Input is 30 coz n_embed = 10 and block_size = 3 i.e. [e,m,m] -> [0.1,0.4,45,.....5.6] this single example will have 30 data point

now let say output is [e,m,m] -> [a] i.e. we can have 27 possible characters thus output as 27 (we have 0 -> '.')

"""


with torch.no_grad():
  # last layer: make less confident coz we dont want that it randomly assign high confidence an character which is not supposed to be (i.e. learning will have to make a lot of effort to correct itself)
  # conclusion: we can have smooth start in training
  model.layers[-1].weight *= 0.1

  # all other layers: apply gain (as Tanh() try to squash them overtime, i.e. std will come near to 0, so mathmatically calculated gain is multiplied)
  for layer in model.layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *=  5/3

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
  p.requires_grad = True

Xtr.shape, Xtr.shape[0]

#  same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []
# update to data ratio
ud = [] # enabled here because the later update-ratio plot needs it

for i in range(max_steps):

  # minibatch construct
  ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
  Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

  # forward pass
  # emb = C[Xb] # embed the characters into vectors
  # x = emb.view(emb.shape[0], -1) # concatenate the vectors
  # x = Xb
  # for layer in layers:
  #   x = layer(x)
  logits = model(Xb)
  loss = F.cross_entropy(logits, Yb) # loss function

  # needed below by the backward-pass histogram cell
  if i == max_steps - 1:
    for layer in model.layers:
      layer.out.retain_grad()
  for p in parameters:
    p.grad = None
  loss.backward()

  # update
  lr = 0.1 if i < 150000 else 0.01 # step learning rate decay
  for p in parameters:
    assert p.grad is not None
    p.data += -lr * p.grad

  # track stats
  if i % 1000 == 0: # print every once in a while
    print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
  lossi.append(loss.log10().item())
  with torch.no_grad():
    ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in parameters])

  # if i >= 10000:
  #   break # AFTER_DEBUG: would take out obviously to run full optimization

for layer in model.layers:
  print(layer.__class__.__name__, ':', tuple(layer.out.shape))

for layer in model.layers:
    if isinstance(layer, FlattenConsecutive):
        print("Flatten n =", layer.n)

plt.plot(torch.tensor(lossi).view(-1,1000).mean(1))
plt.savefig(Path(__file__).resolve().parent / "loss_curve_smoothed.svg", format="svg", bbox_inches="tight")
plt.close()

@torch.no_grad()
def split_loss(split):
    x, y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')
split_loss('test')

# sample from the model

g = torch.Generator().manual_seed(2147483647)

# Put BatchNorm layers in inference mode
for layer in model.layers:
    if isinstance(layer, BatchNorm1d):
        layer.training = False

samples = []
for _ in range(20):

    out = []
    context = [0] * block_size  # initialize with all ...

    while True:
        # character IDs → embeddings
        # emb = C[torch.tensor([context])]
        # # concatenate embeddings
        # x = emb.view(1, -1)
        # # pass through the SAME network we trained
        # for layer in layers:
        #     x = layer(x)
        # # x is now the logits
        # logits = x
        logits = model(torch.tensor([context]))
        # logits → probabilities
        probs = F.softmax(logits, dim=1)
        # ix = probs.argmax().item()
        # sample next character
        ix = int(torch.multinomial(
            probs,
            num_samples=1,
            generator=g
        ).item())
        # move context window forward
        context = context[1:] + [ix]
        # save generated character
        out.append(ix)
        # 0 = '.'
        if ix == 0:
            break
    sample = ''.join(itos[i] for i in out)
    samples.append(sample)
    print(sample)

plt.plot(lossi)
plt.savefig(Path(__file__).resolve().parent / "loss_curve.svg", format="svg", bbox_inches="tight")
plt.close()

# visualize forward pass histograms  ( this one help us to see in the inner tanh layers wheather they are saturated(dead) or non-saturated, high saturation will kill the neurons as there will be
# no change in weight/biases during backpropogation; so through this tool we aim to fix the saturation of out network)
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(model.layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends);
plt.title('activation distribution')
plt.savefig(Path(__file__).resolve().parent / "activations.svg", format="svg", bbox_inches="tight")
plt.close()

# visualize backword pass histograms (this one help us to figure out that our gradients are travelling roughly at same pattern thorought out the layers, as diffrent setting may lead to die out
# of these gradient / or explode it)
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(model.layers[:-1]): # note: exclude the output layer
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends);
plt.title('gradient distribution')
plt.savefig(Path(__file__).resolve().parent / "gradients.svg", format="svg", bbox_inches="tight")
plt.close()

# visualize histograms (# this can usefull for observing the learning rate [how much we are adjusting the weights, we have to figure out best learning rate )
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(parameters):
  t = p.grad
  if p.ndim == 2:
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution');
plt.savefig(Path(__file__).resolve().parent / "weight_gradients.svg", format="svg", bbox_inches="tight")
plt.close()

# this can usefull for observing the learning rate [how much we are adjusting the weights, we have to figure out best learning rate )
plt.figure(figsize=(20, 4))
legends = []
for i,p in enumerate(parameters):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
plt.legend(legends);
plt.savefig(Path(__file__).resolve().parent / "update_ratios.svg", format="svg", bbox_inches="tight")
plt.close()

plt.plot(lossi)
plt.savefig(Path(__file__).resolve().parent / "loss_curve_raw.svg", format="svg", bbox_inches="tight")
plt.close()

# visualize saturation of Tanh activations

with torch.no_grad():

    # take a batch
    ix = torch.randint(0, Xtr.shape[0], (200,), generator=g)
    Xb = Xtr[ix]

    # embeddings
    x = model.layers[0](Xb)

    # flatten
    # x = emb.view(emb.shape[0], -1)

    for i, layer in enumerate(model.layers[1:]):

        x = layer(x)

        if isinstance(layer, Tanh):
            t = layer.out.detach()

            print(
                f"layer {i}: shape={tuple(t.shape)}, "
                f"saturated={(t.abs() > 0.99).float().mean() * 100:.2f}%"
            )

            plt.figure(figsize=(20, 5))
            plt.imshow(
                t.abs() > 0.99,
                cmap="gray",
                interpolation="nearest",
                aspect="auto"
            )
            plt.title(f"Tanh layer {i} saturation")
            plt.xlabel("neurons")
            plt.ylabel("examples")
            plt.savefig(Path(__file__).resolve().parent / f"saturation_layer_{i}.svg", format="svg", bbox_inches="tight")
            plt.close()

plt.plot(lossi)
plt.savefig(Path(__file__).resolve().parent / "loss_curve_final.svg", format="svg", bbox_inches="tight")
plt.close()

# visualize dimensions 0 and 1 of the embedding matrix C for all characters
C = model.layers[0].weight # C is the Embedding weight in this Sequential model
plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid(True, which='minor')
plt.savefig(Path(__file__).resolve().parent / "embeddings.svg", format="svg", bbox_inches="tight")
plt.close()

with open(Path(__file__).resolve().parent / "wavenet_samples.txt", "w") as f:
    f.write("# makemore part 5 sampled names\n")
    f.write(f"# block_size={block_size} n_embed={n_embed} n_hidden={n_hidden} steps={max_steps}\n\n")
    for sample in samples:
        f.write(sample + "\n")
