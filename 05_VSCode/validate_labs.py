"""Validate notebook schema, local links, and Python/shell example syntax; no execution."""
from __future__ import annotations

import ast
from pathlib import Path
import re
import subprocess

import nbformat

HERE = Path(__file__).resolve().parent


def main():
    notebooks = sorted(HERE.glob('05*.ipynb'))
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type != 'markdown':
                raise ValueError(f'{path.name}: these are guided Markdown terminal labs')
            for href in re.findall(r'\]\(([^)]+)\)', cell.source):
                href = href.split('#')[0]
                if href and not re.match(r'[a-z]+://', href) and not (path.parent / href).exists():
                    raise ValueError(f'{path.name}: missing local link {href}')
            for language, code in re.findall(r'^```(python|bash)\n(.*?)^```', cell.source, flags=re.M | re.S):
                if language == 'python':
                    ast.parse(code, filename=path.name)
                else:
                    subprocess.run(['bash', '-n'], input=code, text=True, check=True)
        print(f'Validated {path.name}: {len(notebook.cells)} Markdown cells')
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print('All companion Python files and fenced Python/bash examples parse.')


if __name__ == '__main__':
    main()
