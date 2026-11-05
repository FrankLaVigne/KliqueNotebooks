"""Execute all follow-on notebook cells offline in separate Python processes.

Optional --engine kernel uses local Jupyter IPC. The network guard is a regression
check, not a security sandbox. Source notebooks and their saved outputs stay clean.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OFFLINE_GUARD = '''
import socket
import urllib.request

def _blocked_network(*args, **kwargs):
    raise RuntimeError("Offline validation: external network calls are disabled")

socket.create_connection = _blocked_network
socket.socket.connect = _blocked_network
urllib.request.urlopen = _blocked_network
'''


def execute_python(path: Path, review_dir: Path | None) -> None:
    """Run trusted repository code in order, without pretending it is a kernel."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    namespace = {'__name__': '__main__', 'display': display}
    exec(compile(OFFLINE_GUARD, '<offline-guard>', 'exec'), namespace)  # noqa: S102
    notebook = nbformat.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        exec(compile(cell.source, f'{path.name}:cell-{index + 1}', 'exec'), namespace)  # noqa: S102
        if review_dir:
            for figure_id in plt.get_fignums():
                plt.figure(figure_id).savefig(review_dir / f'{path.stem}-cell-{index + 1}-fig-{figure_id}.png',
                                             bbox_inches='tight')
        plt.close('all')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cwd', choices=['root', 'section'], default='section')
    parser.add_argument('--engine', choices=['python', 'kernel'], default='python')
    parser.add_argument('--output-dir', type=Path, help='Optional review logs/plots or executed kernel copies')
    parser.add_argument('--single-notebook', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir = args.output_dir.resolve()
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.single_notebook:
        execute_python(args.single_notebook, args.output_dir)
        return
    with tempfile.TemporaryDirectory(prefix='klique-notebooks-') as temp:
        work = Path(temp)
        repo = work / 'repo'
        shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(
            '.git', '.venv', 'venv', '__pycache__', 'lab_outputs', '.env*',
            '.ipynb_checkpoints', '.pytest_cache', '.ruff_cache'))
        spec = work / 'jupyter' / 'kernels' / 'klique-offline'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Klique offline validation', 'language': 'python'}))
        os.environ['JUPYTER_PATH'] = str(work / 'jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        os.environ['JUPYTER_RUNTIME_DIR'] = str(work / 'runtime')
        os.environ['IPYTHONDIR'] = str(work / 'ipython')
        os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
        os.environ['MPLBACKEND'] = 'Agg'
        os.environ['XDG_CACHE_HOME'] = str(work / 'cache')
        notebooks = sorted((repo / '05_Jupyter').glob('05[b-n]_*.ipynb'))
        if len(notebooks) != 13:
            raise RuntimeError(f'Expected 13 follow-on notebooks; found {len(notebooks)}')
        for path in notebooks:
            shutil.rmtree(repo / '05_Jupyter' / 'lab_outputs', ignore_errors=True)
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            cwd = repo if args.cwd == 'root' else path.parent
            if args.engine == 'python':
                command = [sys.executable, str(Path(__file__).resolve()), '--single-notebook', str(path)]
                if args.output_dir:
                    command += ['--output-dir', str(args.output_dir)]
                result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=180, check=False)
                if args.output_dir:
                    (args.output_dir / (path.stem + '.log')).write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f'{path.name} failed:\n{result.stdout}\n{result.stderr}')
            else:
                from jupyter_client import KernelManager
                from nbclient import NotebookClient

                notebook.cells.insert(0, nbformat.v4.new_code_cell(OFFLINE_GUARD))
                manager = KernelManager(kernel_name='klique-offline', transport='ipc',
                                        ip=str(work / ('kernel-' + path.stem)))
                NotebookClient(notebook, km=manager, kernel_name='klique-offline', timeout=180,
                               resources={'metadata': {'path': str(cwd)}}).execute()
                notebook.cells.pop(0)
                if args.output_dir:
                    nbformat.write(notebook, args.output_dir / path.name)
            print(f'Passed fresh-{args.engine} offline execution ({args.cwd}): {path.name}', flush=True)
        print(f'Passed {len(notebooks)} notebooks. Optional live/download/CUDA paths were not run.')
        if args.engine == 'python':
            print('Python cell execution is not Jupyter kernel or widget/frontend verification.')


if __name__ == '__main__':
    main()
