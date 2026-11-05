# Data sources, attribution, and download choices

The repository's [MIT license](../LICENSE) covers its original teaching code. Dataset and documentation rights are separate. No downloaded image corpus is committed here. Review the linked terms before use or redistribution, especially derived datasets. Public availability is not permission to ignore licensing.

| Data | Attribution and primary source | Terms and size | Default behavior |
|---|---|---|---|
| MNIST | Yann LeCun, Corinna Cortes, Christopher J. C. Burges; [original dataset description](https://yann.lecun.com/exdb/mnist/), [Keras dataset documentation](https://keras.io/api/datasets/mnist/) | Keras lists CC BY-SA 3.0. 60,000 training and 10,000 test 28×28 images; four gzip files total about 11.6 MB | Explicit `allow_download=True`, otherwise verified cache or clearly named sklearn digits fallback |
| Offline sklearn digits | E. Alpaydin and C. Kaynak, [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits); [sklearn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | UCI lists CC BY 4.0. sklearn bundles 1,797 8×8 images. Not MNIST | No download; deterministic stratified 75/25 split, then bounded sampling |
| EMNIST | Gregory Cohen, Saeed Afshar, Jonathan Tapson, André van Schaik; *EMNIST: an extension of MNIST to handwritten letters*, 2017; [NIST dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset), [paper](https://arxiv.org/abs/1702.05373), [Western Sydney dataset record](https://doi.org/10.26183/zn7s-gh79) | NIST-derived data is not automatically public domain. Western Sydney's record lists CC BY-ND 4.0; review those terms before redistribution of transformed data. Combined archive about 562 MB, with several GB of expanded files | Optional compatible torch/torchvision; both `allow_download=True` **and** `allow_emnist_download=True` required. Cached data may be used offline; otherwise 5d performs only a clearly synthetic orientation/mapping exercise |
| Fashion-MNIST | Han Xiao, Kashif Rasul, Roland Vollgraf, Zalando Research; [official repository](https://github.com/zalandoresearch/fashion-mnist), [paper](https://arxiv.org/abs/1708.07747) | [MIT license](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE). Four gzip files total about 30 MB; 60,000/10,000 images | Explicit download, otherwise verified cache or honestly labeled digits fallback, not a clothing result |
| Bike Sharing | Hadi Fanaee-T and João Gama; *Event labeling combining ensemble detectors and background knowledge*, 2013; [UCI dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset); original [Capital Bikeshare system data](https://capitalbikeshare.com/system-data) | UCI lists CC BY 4.0. The UCI ZIP is about 280 KB, containing 2011–2012 day/hour aggregates. This lab uses `day.csv`, not current trip-level downloads | Synthetic seasonal fixture by default. `USE_REAL_DATA=True` in 5k permits a bounded download of the fixed UCI archive |
| Telemetry and messy CSV | Invented in the lesson code | Original teaching fixtures under repository MIT; no real customers, devices, or operations | Generated locally with fixed seeds / embedded strings |
| Public document RAG | This repository's root README and section README only | Original repository documentation under MIT. The allowlist does not crawl arbitrary files or fetch websites | Local retrieval and extractive answers; gateway generation off by default |

## Image cache and verification

`image_lab.py` uses the HTTPS MNIST mirror used by [torchvision's MNIST implementation](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py), and the official Zalando Fashion-MNIST repository's gzip files. It checks the published MD5 hashes of compressed bytes, a 40 MB per-file download cap, a 60 MB expanded IDX cap, expected dimensions/counts, and label range. MD5 is a corruption check, **not** a modern authenticity guarantee. Never disable TLS certificate verification to make a download work.

Images are cached under `05_Jupyter/lab_outputs/data/` when using the notebooks. Do not put datasets or generated outputs in Git. The loader preserves official train/test boundaries before sampling, normalizes image pixels to [0,1], and records the requested and actual dataset plus split provenance. Missing cache plus disabled download falls back only for MNIST/Fashion, with a visible warning. Failed enabled downloads raise an error rather than quietly changing the dataset.

EMNIST intentionally uses the separately installed torchvision adapter and its upstream archive checksum. It downloads the complete archive and has no small-subset download guarantee; a sample limit reduces training compute, not archive size. Use a pre-staged verified cache when possible. With torchvision installed, an explicit real-data call is:

```python
emnist = load_image_data(
    "emnist_letters", cache_dir=OUTPUT / "data",
    allow_download=True, allow_emnist_download=True,
    limit_train=3000, limit_test=600, seed=42,
)
```

EMNIST `letters` maps original labels 1–26 to 0–25. `balanced` uses 47 digit/letter classes with merged case variants, not 62 independent characters. Raw EMNIST images are transposed into upright orientation before modeling. Inspect class names and samples rather than applying the letters mapping to balanced data.

## Resource limits and offline interpretation

The teaching defaults cap image samples at a few thousand, logistic-regression iterations at 100, search candidates at three, matrix sizes at 512, and anomaly records at a small synthetic series. Runtime depends on CPU, BLAS threads, and package versions. CPU execution needs no model-serving endpoint. Optional PyTorch/GPU, ClearML, real downloads, live gateway answers, and hosted workspace restoration are distinct checks, not implied by passing offline tests.

The offline fixtures test workflow and software behavior. Their scores do not establish MNIST/EMNIST/Fashion performance, actual Capital Bikeshare forecasts, production anomaly detection quality, or live cloud connectivity. Keep actual data names with all results and do not compare fallback metrics with full-data benchmark numbers.
