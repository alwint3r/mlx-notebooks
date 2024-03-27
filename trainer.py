import mlx.nn as nn
import mlx.data as dx
import mlx.optimizers as optim
import mlx.core as mx
import time

from functools import partial

def train_epoch(model: nn.Module, data: dx.Stream, optimizer: optim.Optimizer, epoch_n: int, verbose: bool = True):
  def train_step(model, X, y):
    logits = model(X)
    softmax = nn.softmax(logits)
    loss = mx.mean(nn.losses.cross_entropy(softmax, y))
    accuracy = mx.mean(mx.argmax(softmax, axis=1) == y)
    return loss, accuracy
  
  losses = []
  accuracies = []
  samples_per_seconds = []
  
  state = [model.state, optimizer.state]

  @partial(mx.compile, inputs=state, outputs=state)
  def step(X, y):
    train_step_fn = nn.value_and_grad(model, train_step)
    (loss, accuracy), grads = train_step_fn(model, X, y)
    optimizer.update(model, grads)
    return loss, accuracy

  for batch_idx, batch in enumerate(data):
    X, y = mx.array(batch["image"]), mx.array(batch["label"])
    tic = time.perf_counter()
    loss, accuracy = step(X, y)
    mx.eval(state)
    toc = time.perf_counter()

    loss = loss.item()
    accuracy = accuracy.item()
    losses.append(loss)
    accuracies.append(accuracy)
    throughput = X.shape[0] / (toc - tic)
    samples_per_seconds.append(throughput)

    if (batch_idx % 10 == 0) and verbose:
      print(" | ".join((
                f"Epoch {epoch_n:02d} [{batch_idx:03d}]",
                f"Train loss {loss:.3f}",
                f"Train acc {accuracy:.3f}",
                f"Throughput {throughput:.2f} images/second",
            )))

  mean_train_loss = mx.mean(mx.array(losses))
  mean_train_acc = mx.mean(mx.array(accuracies))
  mean_samples_per_second = mx.mean(mx.array(samples_per_seconds))
  return mean_train_loss, mean_train_acc, mean_samples_per_second

def test_epoch(model: nn.Module, data: dx.Stream, epoch_n: int):
  accuracies = []
  for batch in data:
    X, y = mx.array(batch["image"]), mx.array(batch["label"])
    logits = model(X)
    softmax = nn.softmax(logits)
    accuracy = mx.mean(mx.argmax(softmax, axis=1) == y)
    accuracies.append(accuracy)
  return mx.mean(mx.array(accuracies))
