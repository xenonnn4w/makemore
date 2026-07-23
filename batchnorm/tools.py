"""Training diagnostics for a deep MLP.

Four views of the same question: is this network trainable, or is it quietly
broken? Each function prints a table, saves an SVG, and returns its report
lines so the caller can also write them to a log file.

Call them right after loss.backward() on a step where layer.out.retain_grad()
was set, otherwise the gradient plots have nothing to read.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from layers import Linear, Tanh

HERE = Path(__file__).resolve().parent


def _save(fig_title, out_name, legends):
  plt.legend(legends)
  plt.title(fig_title)
  plt.savefig(HERE / out_name, format="svg", bbox_inches="tight")
  plt.close()
  return f"wrote {out_name}"


def activation_stats(layers, out_name="activations.svg"):
  """Distribution of each Tanh output.

  What to look for: saturation. A tanh pinned near +/-1 has a near-zero local
  gradient, so the neurons behind it stop learning - they are dead. Anything
  much above ~5% saturated in the deeper layers means the init gain is too hot.
  """
  lines = ["activation distribution (Tanh outputs)"]
  plt.figure(figsize=(20, 4))
  legends = []
  for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
    if isinstance(layer, Tanh):
      t = layer.out.detach() # stats only, keep autograd out of it
      lines.append('  layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (
        i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean() * 100))
      hy, hx = torch.histogram(t, density=True)
      plt.plot(hx[:-1], hy)
      legends.append(f'layer {i} ({layer.__class__.__name__})')
  lines.append("  " + _save("activation distribution", out_name, legends))
  return lines


def gradient_stats(layers, out_name="gradients.svg"):
  """Distribution of the gradient arriving at each Tanh output.

  What to look for: the spread should stay roughly constant across depth. A std
  that shrinks layer by layer is a vanishing gradient; one that grows is an
  exploding gradient. Either way the deep layers train at a different speed
  than the shallow ones.
  """
  lines = ["gradient distribution (Tanh outputs)"]
  plt.figure(figsize=(20, 4))
  legends = []
  for i, layer in enumerate(layers[:-1]): # note: exclude the output layer
    if isinstance(layer, Tanh):
      if layer.out.grad is None: # retain_grad() was not set on this step
        lines.append(f'  layer {i}: no gradient retained, skipped')
        continue
      t = layer.out.grad.detach()
      lines.append('  layer %d (%10s): mean %+f, std %e' % (
        i, layer.__class__.__name__, t.mean(), t.std()))
      hy, hx = torch.histogram(t, density=True)
      plt.plot(hx[:-1], hy)
      legends.append(f'layer {i} ({layer.__class__.__name__})')
  lines.append("  " + _save("gradient distribution", out_name, legends))
  return lines


def weight_gradient_stats(parameters, out_name="weight_gradients.svg"):
  """Gradient distribution per weight matrix, next to the weights themselves.

  What to look for: the grad:data ratio. A weight matrix whose gradients are
  tiny relative to its values barely moves; one where they are comparable is
  being rewritten every step. The last layer usually stands out here.
  """
  lines = ["weight gradient distribution"]
  plt.figure(figsize=(20, 4))
  legends = []
  for i, p in enumerate(parameters):
    if p.ndim == 2 and p.grad is not None:
      t = p.grad.detach()
      lines.append('  weight %10s | mean %+f | std %e | grad:data ratio %e' % (
        tuple(p.shape), t.mean(), t.std(), t.std() / p.detach().std()))
      hy, hx = torch.histogram(t, density=True)
      plt.plot(hx[:-1], hy)
      legends.append(f'{i} {tuple(p.shape)}')
  lines.append("  " + _save("weights gradient distribution", out_name, legends))
  return lines


def update_ratio_plot(parameters, ud, out_name="update_ratios.svg"):
  """log10 of (update size / weight size) per parameter, over training.

  What to look for: every curve hugging the black -3 line, i.e. each step
  changes a weight by about 0.1% of its own scale. Well above -3 means the
  learning rate is too high; well below means too low and training crawls.
  """
  lines = ["update-to-data ratios (log10, target ~ -3)"]
  plt.figure(figsize=(20, 4))
  legends = []
  for i, p in enumerate(parameters):
    if p.ndim == 2:
      plt.plot([ud[j][i] for j in range(len(ud))])
      legends.append('param %d %s' % (i, tuple(p.shape)))
      lines.append('  param %d %10s | final ratio %+.2f' % (i, tuple(p.shape), ud[-1][i]))
  plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3, indicate on plot
  lines.append("  " + _save("update-to-data ratios", out_name, legends))
  return lines


def loss_curve(lossi, out_name="loss_curve.svg"):
  """Minibatch loss over training, on a log10 scale."""
  plt.figure(figsize=(10, 5))
  plt.plot(lossi, linewidth=0.5)
  plt.xlabel("step")
  plt.ylabel("log10(minibatch cross-entropy loss)")
  return ["training loss", "  " + _save("training loss", out_name, [])]


def report_all(layers, parameters, ud, lossi):
  """Run every diagnostic, print it, and return the full report as lines."""
  lines = []
  for block in (activation_stats(layers),
                gradient_stats(layers),
                weight_gradient_stats(parameters),
                update_ratio_plot(parameters, ud),
                loss_curve(lossi)):
    lines.extend(block)
    lines.append("")
  for line in lines:
    print(line)
  return lines
