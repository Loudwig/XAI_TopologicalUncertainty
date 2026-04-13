
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch



CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_cifar10_datasets(root: str = "./data"):
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# Generate gaussian noise image
# It is our OOD dataset
class GaussianNoiseDataset(Dataset):
    """
    raw pixels  N(0.5, 0.1), clipped to [0, 1] (same as in the paper)
    then normalized with CIFAR-10 mean/std before entering the network.
    """
    def __init__(self, n_samples: int):
        self.n_samples = int(n_samples)
        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.normal(mean=0.5, std=0.1, size=(3, 32, 32)).clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        y = -1
        return x, y


def make_loader(dataset, batch_size: int = 128, shuffle: bool = False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

# A small CNN that is like the one used in the paper
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

# load a pretrained network trained on CIFAR-10
def load_pretrained_resnet56(device: str = "cpu"):
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        "cifar10_resnet56",
        pretrained=True,
    )
    model.eval()
    model.to(device)
    return model


def eval_accuracy(model, data_loader, device: str = "cpu") -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / max(total, 1)

# function to train our CNN
def train_model(model, train_loader, test_loader, epochs: int = 10, device: str = "cpu"):
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        test_acc = eval_accuracy(model, test_loader, device=device)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}"
        )

# Register forward hooks to store intermediate feature maps used for TU.
class ActivationRecorder:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = list(layer_names)
        self.activations = {}
        self.handles = []

        modules = dict(model.named_modules())
        for name in self.layer_names:
            if name not in modules:
                raise ValueError(f"Layer {name!r} not found in model.")
            handle = modules[name].register_forward_hook(self._hook(name))
            self.handles.append(handle)

    def _hook(self, name):
        def fn(module, inputs, output):
            self.activations[name] = output.detach()
        return fn

    def clear(self):
        self.activations = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _channel_distance_matrix(feature_map: torch.Tensor) -> np.ndarray:
    x = feature_map.detach().float().cpu().numpy()
    c = x.shape[0]
    x = x.reshape(c, -1)

    if x.shape[1] <= 1:
        return np.zeros((c, c), dtype=np.float32)

    corr = np.corrcoef(x)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    dist = 1.0 - np.abs(corr)
    dist = np.asarray(dist, dtype=np.float32)
    dist = 0.5 * (dist + dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist


def _mst_edge_weights(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Prim's algorithm on the dense distance matrix.
    Returns the sorted MST edge weights.
    """
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.zeros(1, dtype=np.float32)

    selected = np.zeros(n, dtype=bool)
    selected[0] = True
    best = distance_matrix[0].copy()
    best[0] = np.inf
    edges = []

    for _ in range(n - 1):
        j = int(np.argmin(best))
        edges.append(best[j])
        selected[j] = True
        best[j] = np.inf

        row = distance_matrix[j]
        for k in range(n):
            if not selected[k] and row[k] < best[k]:
                best[k] = row[k]

    return np.sort(np.asarray(edges, dtype=np.float32))


def feature_map_to_topological_vector(feature_map: torch.Tensor) -> np.ndarray:
    dist = _channel_distance_matrix(feature_map)
    return _mst_edge_weights(dist)

# For each predicted class we compute an average topological reference
# over CIFAR-10 training samples
def build_class_references(
    model,
    train_loader,
    layer_names,
    max_per_class: int,
    device: str = "cpu",
):
    recorder = ActivationRecorder(model, layer_names)
    num_classes = 10

    sums = {layer: [None for _ in range(num_classes)] for layer in layer_names}
    counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, labels in train_loader:
            if np.all(counts >= max_per_class):
                break

            images = images.to(device)
            labels = labels.to(device)

            recorder.clear()
            _ = model(images)

            for b in range(images.shape[0]):
                cls = int(labels[b].item())
                if counts[cls] >= max_per_class:
                    continue

                for layer in layer_names:
                    vec = feature_map_to_topological_vector(recorder.activations[layer][b])
                    if sums[layer][cls] is None:
                        sums[layer][cls] = np.zeros_like(vec, dtype=np.float64)
                    sums[layer][cls] += vec

                counts[cls] += 1

    references = {layer: {} for layer in layer_names}
    for layer in layer_names:
        for cls in range(num_classes):
            if sums[layer][cls] is None or counts[cls] == 0:
                raise RuntimeError(f"No reference built for class {cls} at layer {layer}.")
            references[layer][cls] = (sums[layer][cls] / counts[cls]).astype(np.float32)

    recorder.remove()
    return references

# TU is the weighted average distance between the sample signature
# and the class reference signatures across selected layers
def _topological_score(sample_vectors, pred_class: int, references, layer_weights):
    num = 0.0
    den = 0.0

    for layer, weight in layer_weights.items():
        if weight == 0:
            continue
        ref = references[layer][pred_class]
        vec = sample_vectors[layer]
        score = np.mean(np.abs(vec - ref))
        num += float(weight) * float(score)
        den += float(weight)

    if den == 0:
        raise ValueError("All layer weights are zero.")

    return num / den

# Standard confidence baseline: high score means more likely OOD
# We use 1 - max softmax probability
def score_dataset(model, data_loader, references, layer_weights, device: str = "cpu"):
    layer_names = list(layer_weights.keys())
    recorder = ActivationRecorder(model, layer_names)

    baseline_scores = []
    tu_scores = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)

            recorder.clear()
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            confs, preds = probs.max(dim=1)
            baseline_batch = (1.0 - confs).detach().cpu().numpy()

            for i in range(images.shape[0]):
                baseline_scores.append(float(baseline_batch[i]))
                sample_vectors = {
                    layer: feature_map_to_topological_vector(recorder.activations[layer][i])
                    for layer in layer_names
                }
                tu = _topological_score(
                    sample_vectors=sample_vectors,
                    pred_class=int(preds[i].item()),
                    references=references,
                    layer_weights=layer_weights,
                )
                tu_scores.append(float(tu))

    recorder.remove()
    return np.asarray(baseline_scores), np.asarray(tu_scores)


def roc_curve_from_scores(id_scores, ood_scores):
    id_scores = np.asarray(id_scores, dtype=np.float64)
    ood_scores = np.asarray(ood_scores, dtype=np.float64)

    scores = np.concatenate([id_scores, ood_scores], axis=0)
    labels = np.concatenate(
        [
            np.zeros(len(id_scores), dtype=np.int64),
            np.ones(len(ood_scores), dtype=np.int64),
        ],
        axis=0,
    )

    order = np.argsort(scores)[::-1]
    scores = scores[order]
    labels = labels[order]

    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    change = np.r_[True, scores[1:] != scores[:-1]]
    tps = tps[change]
    fps = fps[change]

    tpr = np.r_[0.0, tps / max(pos, 1), 1.0]
    fpr = np.r_[0.0, fps / max(neg, 1), 1.0]
    return fpr, tpr


def auc_from_roc(fpr, tpr):
    fpr = np.asarray(fpr, dtype=np.float64)
    tpr = np.asarray(tpr, dtype=np.float64)
    return float(np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr))
