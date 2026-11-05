"""Small, offline-first image labs. Downloads are explicit; test splits stay separate.

MNIST mirror and checksums: torchvision.datasets.MNIST resources.
Fashion-MNIST files/checksums: github.com/zalandoresearch/fashion-mnist.
MD5 detects corruption against published checksums, not adversarial authenticity;
HTTPS is required. EMNIST uses torchvision's verified NIST archive adapter.
"""
from dataclasses import dataclass
from pathlib import Path
from numbers import Integral, Real
import gzip
import hashlib
import struct
import time
import urllib.request
from urllib.parse import urlsplit
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class ImageData:
    train_images: np.ndarray
    train_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    class_names: list
    name: str
    source: str
    provenance: dict


class DatasetUnavailable(RuntimeError):
    """An optional dataset is not cached or its explicitly enabled fetch failed."""


_FILES = ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
          't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
_HASHES = {
    'mnist': ('f68b3c2dcbeaaa9fbdd348bbdeb94873', 'd53e105ee54ea40749a09fcbcd1e9432',
              '9fb629c4189551a2d022fa330f9573f3', 'ec29112dd5afa0611ce80d1b7f02629c'),
    'fashion_mnist': ('8d4fb7e6c68d591d4c3dfef9ec88bf0d', '25c81989df183df01b3e8a0aad5dffbe',
                      'bef4ecab320f06d8554ea6380940ec79', 'bb300cfdad3c16e7a12a480ee83cd310'),
}
_URLS = {'mnist': 'https://ossci-datasets.s3.amazonaws.com/mnist/',
         'fashion_mnist': 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'}
_FASHION_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                  'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def _subset(images, labels, limit, seed):
    if not isinstance(limit, int) or limit < len(np.unique(labels)):
        raise ValueError('Sample limit must be an integer at least the number of classes.')
    if limit < len(labels):
        # Allocate at least one example/class, then proportional quotas. Unlike
        # train_test_split this also works when the omitted set is very small.
        classes, counts = np.unique(labels, return_counts=True)
        ideal = (limit - len(classes)) * (counts - 1) / (len(labels) - len(classes))
        quotas = 1 + np.floor(ideal).astype(int)
        remaining = limit - int(quotas.sum())
        order = np.argsort(-(ideal - np.floor(ideal)), kind='stable')
        quotas[order[:remaining]] += 1
        rng = np.random.default_rng(seed)
        indices = np.concatenate([rng.choice(np.flatnonzero(labels == value),
                                  size=int(quota), replace=False)
                                  for value, quota in zip(classes, quotas)])
        rng.shuffle(indices)
        return images[indices], labels[indices]
    return images, labels


def _idx(path):
    # The largest official file is 47 MB uncompressed. Bound decompression too.
    with gzip.open(path, 'rb') as stream:
        raw = stream.read(60_000_001)
    if len(raw) > 60_000_000 or len(raw) < 8:
        raise ValueError('IDX file is oversized or truncated.')
    magic, count = struct.unpack('>II', raw[:8])
    if magic == 2051:
        if len(raw) < 16:
            raise ValueError('Truncated IDX image header.')
        rows, cols = struct.unpack('>II', raw[8:16])
        if (rows, cols) != (28, 28) or count not in (60000, 10000):
            raise ValueError('Unexpected IDX image dimensions/count.')
        shape, offset = (count, rows, cols), 16
    elif magic == 2049 and count in (60000, 10000):
        shape, offset = (count,), 8
    else:
        raise ValueError('Unexpected IDX format/count.')
    if len(raw) - offset != int(np.prod(shape)):
        raise ValueError('IDX payload size does not match its header.')
    return np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Dataset download redirects are disabled.')


def _fetch(url, path, checksum):
    parsed = urlsplit(url)
    allowed_hosts = {'ossci-datasets.s3.amazonaws.com', 'raw.githubusercontent.com'}
    known_urls = {base + name for base in _URLS.values() for name in _FILES}
    if (parsed.scheme != 'https' or parsed.hostname not in allowed_hosts
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or url not in known_urls):
        raise ValueError('Dataset URL must be a fixed allowlisted HTTPS resource.')
    opener = urllib.request.build_opener(_NoRedirect())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    try:
        with opener.open(url, timeout=30) as response, temporary.open('wb') as out:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 40_000_000:
                    raise ValueError('Download exceeds the 40 MB per-file safety limit.')
                out.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != checksum:
            raise ValueError('Downloaded dataset checksum mismatch.')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def orient_emnist(images):
    """Transpose raw torchvision/NIST EMNIST pixels into upright display order."""
    images = np.asarray(images)
    if images.ndim not in (2, 3) or images.shape[-2:] != (28, 28):
        raise ValueError('EMNIST images must have shape (28,28) or (N,28,28).')
    return images.swapaxes(-1, -2).copy()


def _emnist(dataset, cache_dir, allow_download, allow_emnist_download):
    if allow_emnist_download and not allow_download:
        raise ValueError('EMNIST download requires both download flags set to True.')
    try:
        from torchvision.datasets import EMNIST
    except (ImportError, RuntimeError) as exc:
        raise DatasetUnavailable('EMNIST needs compatible optional torch/torchvision installations.') from exc
    split = dataset.removeprefix('emnist_')
    try:
        train = EMNIST(str(cache_dir), split=split, train=True,
                       download=bool(allow_download and allow_emnist_download))
        test = EMNIST(str(cache_dir), split=split, train=False, download=False)
    except (RuntimeError, OSError) as exc:
        raise DatasetUnavailable('EMNIST is not available in the local cache. To fetch the NIST '
                                 'archive (~562 MB compressed, more disk space when expanded), '
                                 'set allow_download=True AND allow_emnist_download=True. '
                                 'This optional torchvision path has no small-download guarantee.') from exc
    names = list(train.classes)
    # torchvision's balanced map sorts digits + letters minus 15 merged lowercase
    # forms. Source: docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html
    if split == 'balanced' and names != list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt'):
        raise ValueError('Unexpected torchvision EMNIST balanced class mapping.')
    y_train, y_test = train.targets.numpy(), test.targets.numpy()
    if split == 'letters':
        y_train, y_test = y_train - 1, y_test - 1
        names = list('abcdefghijklmnopqrstuvwxyz')
    return (orient_emnist(train.data.numpy()), y_train,
            orient_emnist(test.data.numpy()), y_test, names)


def load_image_data(dataset='digits', allow_download=False,
                    cache_dir='lab_outputs/data', limit_train=3000, limit_test=600,
                    seed=42, allow_emnist_download=False):
    """Load normalized float32 images, preserving official train/test boundaries.

    Missing MNIST/Fashion caches fall back explicitly to sklearn digits only when
    downloads are disabled. EMNIST never falls back to a fake letter dataset.
    Optional large EMNIST download requires both flags. No network by default.
    """
    if dataset not in ('digits', 'mnist', 'fashion_mnist', 'emnist_letters', 'emnist_balanced'):
        raise ValueError('Unknown dataset: ' + str(dataset))
    requested, cache_dir = dataset, Path(cache_dir)
    if dataset in _HASHES:
        paths = [cache_dir / dataset / name for name in _FILES]
        if not all(path.exists() for path in paths) and not allow_download:
            warnings.warn(f'{dataset} not cached; using sklearn digits (8x8), NOT {dataset}.',
                          RuntimeWarning, stacklevel=2)
            dataset = 'digits'
        else:
            for path, checksum in zip(paths, _HASHES[dataset]):
                if not path.exists():
                    try:
                        _fetch(_URLS[dataset] + path.name, path, checksum)
                    except (OSError, ValueError) as exc:
                        raise DatasetUnavailable(f'Could not fetch {dataset}: {exc}') from exc
                if path.stat().st_size > 40_000_000 or hashlib.md5(path.read_bytes()).hexdigest() != checksum:
                    raise ValueError(f'Cached checksum mismatch: {path}; remove this file and retry explicitly.')
            x_train, y_train, x_test, y_test = map(_idx, paths)
            names = _FASHION_NAMES if dataset == 'fashion_mnist' else list(map(str, range(10)))
            source, scale, split = _URLS[dataset], 255.0, 'official train/test'
    if dataset == 'digits':
        digits = load_digits()
        x_train, x_test, y_train, y_test = train_test_split(
            digits.images, digits.target, test_size=0.25, stratify=digits.target, random_state=seed)
        names, scale = list(map(str, range(10))), 16.0
        source = 'https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html'
        split = 'seeded stratified 75/25 split of bundled sklearn digits (not official MNIST)'
    elif dataset.startswith('emnist_'):
        x_train, y_train, x_test, y_test, names = _emnist(
            dataset, cache_dir, allow_download, allow_emnist_download)
        source, scale, split = 'https://www.nist.gov/itl/products-and-services/emnist-dataset', 255.0, 'official train/test'
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError('Image/label counts differ.')
    for labels in (y_train, y_test):
        if np.min(labels) < 0 or np.max(labels) >= len(names):
            raise ValueError('Dataset labels outside declared class range.')
    x_train, y_train = _subset(x_train, y_train, limit_train, seed)
    x_test, y_test = _subset(x_test, y_test, limit_test, seed)
    return ImageData(x_train.astype(np.float32) / scale, y_train.astype(np.int64),
                     x_test.astype(np.float32) / scale, y_test.astype(np.int64), names,
                     dataset, source, {'requested': requested, 'actual': dataset, 'split': split,
                     'seed': seed, 'normalization': f'pixel / {scale}',
                     'orientation': 'transposed raw EMNIST' if dataset.startswith('emnist') else 'native'})


def train_baseline(data, max_iter=80, seed=42, C=1.0):
    """Fit only train arrays; evaluate the supplied test arrays (or validation fold)."""
    if (isinstance(max_iter, bool) or not isinstance(max_iter, Integral)
            or not 1 <= max_iter <= 1000 or isinstance(C, bool)
            or not isinstance(C, Real) or not np.isfinite(C) or C <= 0):
        raise ValueError('Use an integer 1..1000 iterations and finite C > 0.')
    start = time.perf_counter()
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
    model.fit(data.train_images.reshape(len(data.train_images), -1), data.train_labels)
    seconds = time.perf_counter() - start
    predicted = model.predict(data.test_images.reshape(len(data.test_images), -1))
    return model, {'accuracy': float(accuracy_score(data.test_labels, predicted)),
                   'confusion_matrix': confusion_matrix(data.test_labels, predicted,
                       labels=np.arange(len(data.class_names))).tolist(),
                   'train_seconds': seconds, 'n_train': len(data.train_labels),
                   'n_test': len(data.test_labels), 'dataset': data.name}


def save_checkpoint(model, path):
    """Save only numeric logistic-regression parameters; never pickle a model."""
    if not isinstance(model, LogisticRegression) or not hasattr(model, 'coef_'):
        raise ValueError('Only fitted LogisticRegression checkpoints are supported.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, version=np.array([1]), coef=model.coef_,
                            intercept=model.intercept_, classes=model.classes_)
    return path


def load_checkpoint(path):
    """Restore a bounded numeric NPZ after validating headers before allocation.

    Object arrays, duplicate/unexpected members, oversized headers and mismatched
    declared payload sizes are rejected. No pickle deserialization is enabled.
    """
    import io
    import math
    import zipfile
    # A bounded immutable snapshot avoids validating one file then loading another.
    with Path(path).open('rb') as stream:
        payload = stream.read(21_000_001)
    if len(payload) > 21_000_000:
        raise ValueError('Checkpoint exceeds 21 MB archive limit.')
    buffer = io.BytesIO(payload)
    expected = {'version.npy', 'coef.npy', 'intercept.npy', 'classes.npy'}
    shapes = {}
    with zipfile.ZipFile(buffer) as archive:
        members = archive.infolist()
        if len(members) != len(expected) or {info.filename for info in members} != expected:
            raise ValueError('Duplicate or unexpected checkpoint members.')
        if sum(info.file_size for info in members) > 20_000_000:
            raise ValueError('Checkpoint exceeds 20 MB uncompressed limit.')
        for info in members:
            with archive.open(info) as member:
                format_version = np.lib.format.read_magic(member)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                if format_version not in readers:
                    raise ValueError('Unsupported checkpoint NPY version.')
                shape, _, dtype = readers[format_version](member, max_header_size=4096)
                expected_ndim = 2 if info.filename == 'coef.npy' else 1
                kinds = 'iu' if info.filename in {'version.npy', 'classes.npy'} else 'fi'
                if (len(shape) != expected_ndim or any(n <= 0 for n in shape)
                        or dtype.hasobject or dtype.fields is not None or dtype.kind not in kinds):
                    raise ValueError('Invalid checkpoint header dtype or dimensions.')
                # Python integer product cannot overflow like fixed-width np.prod.
                declared_bytes = math.prod(shape) * dtype.itemsize
                if declared_bytes > 20_000_000 or declared_bytes != info.file_size - member.tell():
                    raise ValueError('Checkpoint header shape does not match bounded payload size.')
                shapes[info.filename] = shape
        rows, features = shapes['coef.npy']
        class_count = shapes['classes.npy'][0]
        if (shapes['version.npy'] != (1,) or class_count < 2
                or rows != (1 if class_count == 2 else class_count)
                or not 1 <= features <= 10000 or shapes['intercept.npy'] != (rows,)):
            raise ValueError('Invalid checkpoint array dimensions.')
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        version, coef, intercept, classes = [saved[k] for k in ('version', 'coef', 'intercept', 'classes')]
    if version.shape != (1,) or version[0] != 1 or coef.ndim != 2:
        raise ValueError('Invalid checkpoint version or coefficient shape.')
    if (classes.ndim != 1 or len(classes) < 2 or classes.dtype.kind not in 'iu'
            or len(np.unique(classes)) != len(classes)
            or coef.shape[0] != (1 if len(classes) == 2 else len(classes))
            or not 1 <= coef.shape[1] <= 10000 or intercept.shape != (coef.shape[0],)
            or coef.dtype.kind not in 'fi' or intercept.dtype.kind not in 'fi'
            or not np.isfinite(coef).all() or not np.isfinite(intercept).all()):
        raise ValueError('Invalid checkpoint numeric arrays.')
    model = LogisticRegression()
    model.coef_, model.intercept_, model.classes_ = coef, intercept, classes
    model.n_features_in_ = coef.shape[1]
    model.n_iter_ = np.array([0])
    return model


def preprocess_drawing(image, size=(8, 8), invert=False):
    """Resize a PIL image/path or uint8/normalized array, with explicit polarity.

    Returns [0,1] float32. This is a demo, not MNIST-equivalent handwriting
    segmentation: crop/center your glyph, and use a dark background/light stroke.
    size is (height,width). No widget dependency is required.
    """
    from PIL import Image, ImageOps
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert('L')
    elif not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.dtype.kind == 'f':
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError('Float drawing arrays must be finite and normalized to [0,1].')
            array = np.rint(array * 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            raise ValueError('Integer drawing arrays must use uint8.')
        image = Image.fromarray(array)
    image = image.convert('L')
    if invert:
        image = ImageOps.invert(image)
    return np.asarray(image.resize((size[1], size[0]), Image.Resampling.LANCZOS), dtype=np.float32) / 255


def predict_drawing(model, image, image_shape=(8, 8), invert=False):
    pixels = preprocess_drawing(image, image_shape, invert)
    return int(model.predict(pixels.reshape(1, -1))[0])


def train_torch(data, device='cpu', epochs=2, batch_size=64, seed=42):
    """Optional bounded torch MLP. CPU default; CUDA must be requested explicitly.

    Reproducibility is best-effort on a fixed device/version, not cross-device
    bitwise equivalence. Includes device synchronization for timing fairness.
    """
    if device not in ('cpu', 'cuda') or not 1 <= epochs <= 20 or not 1 <= batch_size <= 1024:
        raise ValueError('Use cpu/cuda, 1..20 epochs and batch size 1..1024.')
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available; choose device="cpu".')
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(data.train_images.reshape(len(data.train_images), -1))
    y = torch.from_numpy(data.train_labels)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y),
             batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
                                torch.nn.Linear(64, len(data.class_names))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
    if device == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        test = torch.from_numpy(data.test_images.reshape(len(data.test_images), -1)).to(device)
        predicted = model(test).argmax(dim=1).cpu().numpy()
    return model, {'accuracy': float(accuracy_score(data.test_labels, predicted)),
                   'confusion_matrix': confusion_matrix(data.test_labels, predicted,
                       labels=np.arange(len(data.class_names))).tolist(),
                   'train_seconds': seconds, 'device': device, 'epochs': epochs,
                   'n_train': len(y), 'n_test': len(data.test_labels), 'dataset': data.name}


def benchmark_torch(data, devices=('cpu',), **kwargs):
    """Return timing/accuracy per requested device; unavailable CUDA raises clearly."""
    return [train_torch(data, device=device, **kwargs)[1] for device in devices]
