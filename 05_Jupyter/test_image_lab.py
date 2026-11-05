"""Offline regression tests: python -m unittest discover -s 05_Jupyter -p 'test_*.py'."""
import gzip
import json
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import image_lab as lab


class ImageLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = lab.load_image_data(limit_train=300, limit_test=100)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cls.model, cls.metrics = lab.train_baseline(cls.data, max_iter=100)

    def test_deterministic_normalized_stratified_digits(self):
        other = lab.load_image_data(limit_train=300, limit_test=100)
        np.testing.assert_array_equal(self.data.train_images, other.train_images)
        np.testing.assert_array_equal(self.data.test_labels, other.test_labels)
        self.assertEqual(self.data.train_images.shape, (300, 8, 8))
        self.assertEqual(self.data.train_images.dtype, np.float32)
        self.assertGreaterEqual(self.data.train_images.min(), 0)
        self.assertLessEqual(self.data.train_images.max(), 1)
        self.assertEqual(len(np.unique(self.data.train_labels)), 10)
        self.assertEqual(len(np.unique(self.data.test_labels)), 10)
        self.assertEqual(self.data.name, 'digits')

    def test_fallback_honest_and_never_downloads(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(lab, '_fetch') as fetch:
            for name in ('mnist', 'fashion_mnist'):
                with self.assertWarnsRegex(RuntimeWarning, 'NOT ' + name):
                    data = lab.load_image_data(name, cache_dir=folder)
                self.assertEqual(data.name, 'digits')
                self.assertEqual(data.provenance['requested'], name)
            fetch.assert_not_called()

    def test_subset_near_full_size(self):
        x, y = lab._subset(np.arange(30), np.repeat(np.arange(10), 3), 29, 42)
        self.assertEqual(len(x), 29)
        self.assertEqual(len(np.unique(x)), 29)
        self.assertEqual(len(np.unique(y)), 10)

    def test_invalid_dataset_and_limits(self):
        with self.assertRaises(ValueError):
            lab.load_image_data('fake')
        with self.assertRaises(ValueError):
            lab.load_image_data(limit_train=5)
        with self.assertRaises(ValueError):
            lab.train_baseline(self.data, max_iter=1001)

    def test_metrics_json_and_fit_only_training(self):
        self.assertGreater(self.metrics['accuracy'], 0.8)
        self.assertEqual(np.array(self.metrics['confusion_matrix']).sum(), 100)
        json.dumps(self.metrics)
        original_fit = lab.LogisticRegression.fit
        observed = []
        def recorded_fit(model, x, y, *args, **kwargs):
            observed.append((x.copy(), y.copy()))
            return original_fit(model, x, y, *args, **kwargs)
        with patch.object(lab.LogisticRegression, 'fit', recorded_fit), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            lab.train_baseline(self.data, max_iter=5, C=0.5)
        np.testing.assert_array_equal(observed[0][0], self.data.train_images.reshape(300, -1))
        np.testing.assert_array_equal(observed[0][1], self.data.train_labels)

    def test_checkpoint_roundtrip_and_object_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.npz'
            lab.save_checkpoint(self.model, path)
            restored = lab.load_checkpoint(path)
            x = self.data.test_images.reshape(100, -1)
            np.testing.assert_array_equal(self.model.predict(x), restored.predict(x))
            np.testing.assert_allclose(self.model.predict_proba(x), restored.predict_proba(x))
            np.savez(path, version=[1], coef=np.array([object()], dtype=object),
                     intercept=[0], classes=[0, 1])
            with self.assertRaises(ValueError):
                lab.load_checkpoint(path)

    def test_checkpoint_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.npz'
            np.savez(path, version=[1], coef=np.zeros((4, 8)), intercept=np.zeros(4), classes=[0, 1])
            with self.assertRaises(ValueError):
                lab.load_checkpoint(path)

    def test_checkpoint_malicious_shape_rejected_before_numpy_load(self):
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'huge.npz'
            lab.save_checkpoint(self.model, path)
            with zipfile.ZipFile(path) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            header = io.BytesIO()
            np.lib.format.write_array_header_1_0(header, {
                'descr': '<f8', 'fortran_order': False, 'shape': (10, 10**100)})
            entries['coef.npy'] = header.getvalue()
            with zipfile.ZipFile(path, 'w') as archive:
                for name, value in entries.items():
                    archive.writestr(name, value)
            with patch.object(lab.np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'bounded payload'):
                    lab.load_checkpoint(path)
                load.assert_not_called()

    def test_checkpoint_duplicate_member_rejected_before_numpy_load(self):
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'duplicate.npz'
            lab.save_checkpoint(self.model, path)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with zipfile.ZipFile(path, 'a') as archive:
                    archive.writestr('coef.npy', b'anything')
            with patch.object(lab.np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    lab.load_checkpoint(path)
                load.assert_not_called()

    def test_checkpoint_unsupported_npy_version_rejected(self):
        import io
        import zipfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'unsupported.npz'
            lab.save_checkpoint(self.model, path)
            with zipfile.ZipFile(path) as original:
                entries = {name: original.read(name) for name in original.namelist()}
            entries['coef.npy'] = np.lib.format.magic(3, 0)
            with zipfile.ZipFile(path, 'w') as archive:
                for name, value in entries.items():
                    archive.writestr(name, value)
            with patch.object(lab.np, 'load') as load:
                with self.assertRaisesRegex(ValueError, 'Unsupported'):
                    lab.load_checkpoint(path)
                load.assert_not_called()

    def test_drawing_array_path_polarity_and_prediction(self):
        from PIL import Image
        drawing = np.zeros((28, 28), dtype=np.uint8)
        drawing[5:23, 12:16] = 255
        pixels = lab.preprocess_drawing(drawing)
        self.assertEqual(pixels.shape, (8, 8))
        self.assertEqual(pixels.dtype, np.float32)
        np.testing.assert_allclose(lab.preprocess_drawing(np.zeros((8, 8), np.uint8), invert=True), 1)
        np.testing.assert_allclose(lab.preprocess_drawing(np.ones((8, 8), np.float32)), 1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'drawing.png'
            Image.fromarray(drawing).save(path)
            np.testing.assert_array_equal(pixels, lab.preprocess_drawing(path))
        self.assertIn(lab.predict_drawing(self.model, drawing), range(10))
        with self.assertRaises(ValueError):
            lab.preprocess_drawing(np.full((8, 8), np.nan))

    def test_emnist_orientation_fixture_not_benchmark(self):
        raw = np.zeros((2, 28, 28), np.uint8)
        raw[:, 2, 7] = 255
        upright = lab.orient_emnist(raw)
        self.assertEqual(upright[0, 7, 2], 255)
        self.assertEqual(upright[0, 2, 7], 0)
        np.testing.assert_array_equal(lab.orient_emnist(upright), raw)
        with self.assertRaises(ValueError):
            lab.orient_emnist(np.zeros((8, 8)))

    def test_emnist_cached_adapter_labels_and_flags(self):
        calls = []
        class Tensor:
            def __init__(self, array):
                self.array = array
            def numpy(self):
                return self.array
        class FakeEMNIST:
            def __init__(self, root, split, train, download):
                calls.append((split, train, download))
                self.classes = ['N/A'] + list('abcdefghijklmnopqrstuvwxyz')
                self.targets = Tensor(np.tile(np.arange(1, 27), 3))
                images = np.zeros((78, 28, 28), np.uint8)
                images[:, 2, 7] = 255
                self.data = Tensor(images)
        fake = types.ModuleType('torchvision.datasets')
        fake.EMNIST = FakeEMNIST
        with patch.dict('sys.modules', {'torchvision.datasets': fake}):
            data = lab.load_image_data('emnist_letters', limit_train=52, limit_test=52)
            self.assertEqual(calls, [('letters', True, False), ('letters', False, False)])
            self.assertEqual(data.class_names[0], 'a')
            self.assertEqual(data.class_names[-1], 'z')
            self.assertEqual(data.train_labels.min(), 0)
            self.assertEqual(data.train_labels.max(), 25)
            self.assertEqual(data.train_images[0, 7, 2], 1)
            with self.assertRaises(ValueError):
                lab.load_image_data('emnist_letters', allow_emnist_download=True)

    def test_emnist_missing_never_mislabeled_digits(self):
        fake = types.ModuleType('torchvision.datasets')
        def missing(*args, **kwargs):
            raise RuntimeError('Dataset not found')
        fake.EMNIST = missing
        with patch.dict('sys.modules', {'torchvision.datasets': fake}):
            with self.assertRaisesRegex(lab.DatasetUnavailable, '562 MB'):
                lab.load_image_data('emnist_balanced')

    def test_idx_header_and_payload_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.gz'
            with gzip.open(path, 'wb') as out:
                out.write(struct.pack('>II', 2049, 10000) + bytes(10000))
            self.assertEqual(lab._idx(path).shape, (10000,))
            with gzip.open(path, 'wb') as out:
                out.write(struct.pack('>II', 2049, 10000) + bytes(3))
            with self.assertRaises(ValueError):
                lab._idx(path)

    def test_corrupt_cache_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / 'mnist'
            cache.mkdir()
            for name in lab._FILES:
                (cache / name).write_bytes(b'not a verified dataset')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                lab.load_image_data('mnist', cache_dir=folder)

    def test_download_rejects_nonallowlisted_urls_before_io(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(lab.urllib.request, 'build_opener') as build:
            for url in ('http://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
                        'https://example.com/data.gz',
                        'https://raw.githubusercontent.com/other/repo/main/data.gz',
                        'https://user:secret@ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
                        'https://ossci-datasets.s3.amazonaws.com:444/mnist/train-images-idx3-ubyte.gz'):
                with self.subTest(url=url), self.assertRaisesRegex(ValueError, 'HTTPS'):
                    lab._fetch(url, Path(folder) / 'data.gz', 'unused')
            build.assert_not_called()

    def test_download_redirects_are_rejected(self):
        request = lab.urllib.request.Request(lab._URLS['mnist'] + lab._FILES[0])
        handler = lab._NoRedirect()
        for code in (301, 302, 303, 307, 308):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, 'redirects'):
                handler.redirect_request(request, None, code, 'redirect', {},
                                         'https://example.com/data.gz')
        with tempfile.TemporaryDirectory() as folder, patch.object(lab.urllib.request, 'build_opener') as build:
            build.return_value.open.side_effect = ValueError('Dataset download redirects are disabled.')
            path = Path(folder) / 'data.gz'
            with self.assertRaisesRegex(ValueError, 'redirects'):
                lab._fetch(request.full_url, path, 'unused')
            self.assertIsInstance(build.call_args.args[0], lab._NoRedirect)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix('.gz.part').exists())

    def test_baseline_numeric_validation(self):
        for value in (0, -1, float('nan'), float('inf'), True, '1'):
            with self.subTest(C=value), self.assertRaises(ValueError):
                lab.train_baseline(self.data, C=value)
        for value in (0, 1001, 1.5, float('nan'), float('inf'), True, '80'):
            with self.subTest(max_iter=value), self.assertRaises(ValueError):
                lab.train_baseline(self.data, max_iter=value)

    def test_emnist_balanced_mapping_matches_torchvision_source(self):
        # Official torchvision classes_split_dict['balanced'] definition, verified
        # against docs.pytorch.org/vision/main/_modules/torchvision/datasets/mnist.html.
        import string
        # Include all 15 official merged lowercase forms (including lowercase o).
        merged = {'c', 'i', 'j', 'k', 'l', 'm', 'o', 'p', 's', 'u', 'v', 'w', 'x', 'y', 'z'}
        expected = sorted(set(string.digits + string.ascii_letters) - merged)
        self.assertEqual(''.join(expected), '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt')
        class Tensor:
            def __init__(self, array):
                self.array = array
            def numpy(self):
                return self.array
        class FakeEMNIST:
            def __init__(self, *args, **kwargs):
                self.classes = expected
                self.targets = Tensor(np.tile(np.arange(47), 2))
                self.data = Tensor(np.zeros((94, 28, 28), np.uint8))
        fake = types.ModuleType('torchvision.datasets')
        fake.EMNIST = FakeEMNIST
        with patch.dict('sys.modules', {'torchvision.datasets': fake}):
            data = lab.load_image_data('emnist_balanced', limit_train=47, limit_test=47)
        self.assertEqual(data.class_names, expected)
        self.assertEqual(data.class_names[36], 'a')
        self.assertEqual(data.class_names[46], 't')
        self.assertEqual(data.train_labels.max(), 46)

    def test_torch_invalid_options_without_import(self):
        with self.assertRaises(ValueError):
            lab.train_torch(self.data, device='automatic')
        with self.assertRaises(ValueError):
            lab.train_torch(self.data, epochs=100)


if __name__ == '__main__':
    unittest.main()
