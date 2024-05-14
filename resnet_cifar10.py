import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import mlx.core as mx
import mlx.nn as nn
from mlx.data import datasets
import mlx.optimizers as optim
from datasets_utils import cifar10
from mlx.utils import tree_flatten
import trainer


cifar10_train = datasets.load_cifar10(train=True)
cifar10_test = datasets.load_cifar10(train=False)


def get_streamed_data(data, batch_size=0, shuffled=True):
    def transform(x):
        return x.astype("float32") / 255.0

    buffer = data.shuffle() if shuffled else data
    stream = buffer.to_stream()
    stream = stream.key_transform("image", transform)
    stream = stream.batch(batch_size) if batch_size > 0 else stream
    return stream.prefetch(4, 2)


class ShortcutA(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x):
        return mx.pad(
            x[:, ::2, ::2, :],
            pad_width=[(0, 0), (0, 0), (0, 0),
                       (self.dims // 4, self.dims // 4)],
        )


class ResidualBlock(nn.Module):
    def __init__(self, in_dims, dims, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_dims, dims, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(dims)
        self.conv2 = nn.Conv2d(dims, dims, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm(dims)

        self.shortcut = ShortcutA(dims) if stride != 1 else None

    def __call__(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is None:
            out += x
        else:
            out += self.shortcut(x)
        out = nn.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100, num_channels=3, base_dims=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, base_dims, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(base_dims)

        self.layer1 = self._make_layer(base_dims, base_dims, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_dims, base_dims * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_dims * 2, base_dims *2 * 2, num_blocks[2], stride=2)
        self.linear = nn.Linear(128, num_classes)

    def _make_layer(self, in_dims, dims, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_dims, dims, stride))
            in_dims = dims
        return nn.Sequential(*layers)

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def __call__(self, x):
        x = nn.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = mx.mean(x, axis=[1, 2]).reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


model = ResNet([5, 5, 5], num_classes=len(cifar10.labels))
mx.eval(model)

epochs = 10
optimizer = optim.SGD(learning_rate=0.1, momentum=0.9, weight_decay=0.0001)

train_data = get_streamed_data(
    batch_size=128, data=cifar10_train, shuffled=True)
test_data = get_streamed_data(
    batch_size=128, data=cifar10_test, shuffled=False)

train_accuracies = []
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_loss, train_acc, throughput = trainer.train_epoch(
        model, train_data, optimizer, epoch, verbose=False)
    print(" | ".join(
        (
            f"Epoch: {epoch+1}",
            f"avg. Train loss {train_loss.item():.3f}",
            f"avg. Train acc {train_acc.item():.3f}",
            f"Throughput: {throughput.item():.2f} images/sec",
        )))
    test_acc = trainer.test_epoch(model, test_data, epoch)
    print(f"Epoch: {epoch+1} | Test acc {test_acc.item():.3f}")

    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)

    train_data.reset()
    test_data.reset()

# get precision, recall, and f1-score

y_true = []
y_pred = []
model.eval()
for batch in test_data:
    X, y = batch["image"], batch["label"]
    X, y = mx.array(X), mx.array(y)
    logits = model(X)
    prediction = mx.argmax(mx.softmax(logits), axis=1)
    y_true = y_true + y.tolist()
    y_pred = y_pred + prediction.tolist()

y_true = np.array(y_true)
y_pred = np.array(y_pred)

precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

test_data.reset()
