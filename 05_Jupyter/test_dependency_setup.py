"""Check notebook dependency setup without installing packages or using a network."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

SECTION = Path(__file__).resolve().parent
NOTEBOOKS = sorted(SECTION.glob('05[b-n]_*.ipynb'))


def setup_source(path):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cell = next(cell for cell in notebook['cells']
                if cell.get('id', '').endswith('-dependency-setup'))
    return ''.join(cell['source'])


class DependencySetupTests(unittest.TestCase):
    def test_setup_is_consistent_and_before_imports(self):
        self.assertEqual(len(NOTEBOOKS), 13)
        sources = set()
        for path in NOTEBOOKS:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding='utf-8'))
                first_code = next(cell for cell in notebook['cells'] if cell['cell_type'] == 'code')
                self.assertTrue(first_code['id'].endswith('-dependency-setup'))
                self.assertIsNone(first_code['execution_count'])
                self.assertEqual(first_code['outputs'], [])
                sources.add(setup_source(path))
        self.assertEqual(len(sources), 1)

    def test_default_does_not_install_from_either_working_directory(self):
        for path in NOTEBOOKS:
            for cwd in (SECTION.parent, SECTION):
                with self.subTest(notebook=path.name, cwd=cwd.name):
                    with patch('pathlib.Path.cwd', return_value=cwd), \
                         patch('IPython.get_ipython', side_effect=AssertionError('Unexpected installation')), \
                         redirect_stdout(io.StringIO()):
                        namespace = {}
                        exec(compile(setup_source(path), str(path), 'exec'), namespace)  # noqa: S102
                    self.assertFalse(namespace['INSTALL_PACKAGES'])
                    self.assertEqual(namespace['REQUIREMENTS'], SECTION / 'requirements.txt')

    def test_opt_in_uses_pip_magic_and_quotes_paths_with_spaces(self):
        with tempfile.TemporaryDirectory(prefix='klique setup ') as directory:
            root = Path(directory).resolve()
            section = root / '05_Jupyter'
            section.mkdir()
            (section / 'image_lab.py').touch()
            requirements = section / 'requirements.txt'
            requirements.write_text('scikit-learn>=1.5,<2\n', encoding='utf-8')
            source = setup_source(NOTEBOOKS[0]).replace('INSTALL_PACKAGES = False\n',
                                                      'INSTALL_PACKAGES = True\n', 1)
            for cwd in (root, section):
                with self.subTest(cwd=cwd.name):
                    shell = Mock()
                    with patch('pathlib.Path.cwd', return_value=cwd), \
                         patch('IPython.get_ipython', return_value=shell), \
                         redirect_stdout(io.StringIO()):
                        exec(compile(source, 'dependency-setup', 'exec'), {})  # noqa: S102
                    shell.run_line_magic.assert_called_once_with('pip', f'install -r "{requirements}"')

    def test_missing_requirements_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('pathlib.Path.cwd', return_value=Path(directory)), \
             self.assertRaisesRegex(FileNotFoundError, 'repository root or 05_Jupyter'):
            exec(compile(setup_source(NOTEBOOKS[0]), 'dependency-setup', 'exec'), {})  # noqa: S102

    def test_opt_in_outside_ipython_has_actionable_error(self):
        source = setup_source(NOTEBOOKS[0]).replace('INSTALL_PACKAGES = False\n',
                                                  'INSTALL_PACKAGES = True\n', 1)
        with patch('pathlib.Path.cwd', return_value=SECTION), \
             patch('IPython.get_ipython', return_value=None), \
             redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(RuntimeError, "notebook's IPython Python kernel"):
            exec(compile(source, 'dependency-setup', 'exec'), {})  # noqa: S102


if __name__ == '__main__':
    unittest.main()
