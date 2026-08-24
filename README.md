# Makemore

Character-level name generation in PyTorch, built as a progression from a
count-based bigram model to a hierarchical WaveNet-style network. The code uses
small, hand-built neural-network layers to make embeddings, activations,
gradients, normalization, and optimization behavior easy to inspect.

The training corpus is [`names.txt`](names.txt), with one lowercase name per
line. A `.` token marks the start and end of each name.

## Models

| Stage | Context | Main idea | Recorded output |
| --- | ---: | --- | --- |
| [`bigram_count/`](bigram_count/) | 1 character | Count transitions, apply add-one smoothing, and sample from the resulting probabilities | Count matrix and sampled names |
| [`mlp/`](mlp/) | 3 characters | Learn character embeddings and predict the next character with a single hidden-layer MLP | Loss curve, embedding plot, and sampled names |
| [`batchnorm/`](batchnorm/) | 3 characters | Study initialization, activation saturation, and gradient flow in a deeper hand-built MLP | Loss, activation, gradient, and update-ratio diagnostics |
| [`wavenet/`](wavenet/) | 8 characters | Merge adjacent time steps hierarchically with `FlattenConsecutive` layers in a WaveNet-inspired MLP | Raw and smoothed loss curves, activation and gradient diagnostics, and sampled names |

The WaveNet stage is a hierarchical character model inspired by WaveNet's
receptive-field structure; it is not an audio model.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch matplotlib
```

## Run

Run any stage from the repository root:

```bash
python bigram_count/makemore_bigram_count.py
python mlp/makemore_mlp.py
python batchnorm/makemore_batchnorm.py
python wavenet/makemore_wavenet.py
```

Each script reads the bundled dataset and writes plots or text reports beside
its source file. Existing generated outputs are overwritten on rerun. Training
lengths and model sizes are configured in each script; the deeper models take
longer to run. The deep-MLP diagnostic stage defaults to a 1,000-step run with
`DEBUG = True`; set it to `False` for the 200,000-step training run.

## Supporting code

- [`batchnorm/layers.py`](batchnorm/layers.py) contains the hand-built linear,
  batch-normalization, and activation layers used by the deep MLP.
- [`batchnorm/tools.py`](batchnorm/tools.py) produces activation, gradient,
  parameter, and update-ratio diagnostics.
- [`wavenet/layers.py`](wavenet/layers.py) adds embeddings, hierarchical
  flattening, and sequential composition for the WaveNet-style model.

Dataset splits, parameter initialization, and sampling use explicit seeds where
implemented. The checked-in plots and reports capture the most recent recorded
runs; their headers describe the relevant configuration.
