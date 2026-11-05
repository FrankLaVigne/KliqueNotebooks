"""Offline course layout, notebook-schema, local-link and Python syntax checks.

Run from any working directory with Python 3.10+, nbformat and IPython installed.
This does not execute lab cells, contact Klique, or validate cloud UI behavior.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

ROOT = Path(__file__).resolve().parent
EXPECTED_NOTEBOOKS = {
    '00_Orientation/00_Orientation.ipynb',
    '01_vLLM/01_Deploy_a_Model_with_vLLM.ipynb',
    '02_LLMGateway/02a_Use_the_LLM_Gateway.ipynb',
    '02_LLMGateway/02b_Models_and_Routed_Resources.ipynb',
    '02_LLMGateway/02c_Semantic_Routing.ipynb',
    '03_Governance/03_Govern_Usage_Cost_and_Limits.ipynb',
    '04_Agents/04_Agents_on_a_Governed_Endpoint.ipynb',
    '05_Jupyter/05_Running_Jupyter_Lab.ipynb',
    '05_Jupyter/05b_Explore_MNIST.ipynb',
    '05_Jupyter/05c_Train_a_Digit_Classifier.ipynb',
    '05_Jupyter/05d_Explore_EMNIST.ipynb',
    '05_Jupyter/05e_Compare_Experiments.ipynb',
    '05_Jupyter/05f_Draw_a_Digit.ipynb',
    '05_Jupyter/05g_Test_Data_Quality_and_Robustness.ipynb',
    '05_Jupyter/05h_Ask_Public_Documents.ipynb',
    '05_Jupyter/05i_Benchmark_CPU_and_GPU.ipynb',
    '05_Jupyter/05j_Create_an_Experiment_Report.ipynb',
    '05_Jupyter/05k_Forecast_Bike_Demand.ipynb',
    '05_Jupyter/05l_Detect_Telemetry_Anomalies.ipynb',
    '05_Jupyter/05m_Explore_Fashion_Similarity.ipynb',
    '05_Jupyter/05n_Clean_Messy_Data.ipynb',
    '06_SSH/06_Running_an_SSH_Session.ipynb',
    '06_SSH/06b_Install_ComfyUI_in_an_SSH_Session.ipynb',
    '07_VSCode/07_Running_VS_Code.ipynb',
    '07_VSCode/07b_Build_a_Governed_AI_Playground.ipynb',
    '07_VSCode/07c_Chat_with_This_Repository.ipynb',
    '07_VSCode/07d_Debug_an_Agent.ipynb',
    '07_VSCode/07e_Run_a_Model_Bake_Off.ipynb',
    '07_VSCode/07f_Train_Stop_Restore_Continue.ipynb',
    '08_ClaudeCode/08_Running_Claude_Code.ipynb',
    '09_ClaudeDesktop/09_Running_Claude_Desktop.ipynb',
}
# Built in pieces so the validator does not flag its own obsolete-path guard.
OLD_DIRECTORIES = ('05_' + 'VSCode', '07_' + 'ClaudeCode', '08_' + 'ClaudeDesktop')


def check_links(path: Path, source: str) -> int:
    # Screenshot TODOs in HTML comments are intentionally not file links.
    source = re.sub(r'<!--.*?-->', '', source, flags=re.S)
    source = re.sub(r'^```.*?^```', '', source, flags=re.M | re.S)
    hrefs = re.findall(r'\]\(([^)]+)\)', source)
    hrefs += re.findall(r'(?:src|srcset|href)=[\"\x27]([^\"\x27]+)', source)
    count = 0
    for href in hrefs:
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target = path.parent / unquote(parsed.path)
        if not target.exists():
            raise ValueError(f'{path.relative_to(ROOT)}: missing local link {href}')
        count += 1
    return count


def check_cell_syntax(source: str, filename: str) -> None:
    """Parse Python or IPython syntax without executing cells or shell commands."""
    try:
        ast.parse(source, filename=filename)
    except SyntaxError:
        transformed = TransformerManager().transform_cell(source)
        ast.parse(transformed, filename=filename)


def main() -> None:
    notebooks = sorted(p for folder in ROOT.glob('[0-9][0-9]_*') for p in folder.glob('*.ipynb'))
    actual = {str(p.relative_to(ROOT)) for p in notebooks}
    if actual != EXPECTED_NOTEBOOKS:
        raise ValueError(f'Notebook layout mismatch. Missing: {sorted(EXPECTED_NOTEBOOKS - actual)}; '
                         f'unexpected: {sorted(actual - EXPECTED_NOTEBOOKS)}')
    links = 0
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for i, cell in enumerate(notebook.cells, start=1):
            if any(old in cell.source for old in OLD_DIRECTORIES):
                raise ValueError(f'{path.name}, cell {i}: obsolete directory reference')
            if cell.cell_type == 'code':
                check_cell_syntax(cell.source, filename=f'{path.name}:cell-{i}')
            elif cell.cell_type == 'markdown':
                links += check_links(path, cell.source)
        print(f'Validated {path.relative_to(ROOT)}')
    for path in [ROOT / 'README.md', ROOT / '05_Jupyter/README.md', ROOT / '05_Jupyter/DATASETS.md', ROOT / '07_VSCode/README.md']:
        links += check_links(path, path.read_text(encoding='utf-8'))
    python_files = [*ROOT.glob('*.py'), *ROOT.glob('[0-9][0-9]_*/*.py')]
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    print(f'Passed: {len(notebooks)} notebook schemas, {links} local links, '
          f'all code-cell syntax, and {len(python_files)} Python files. No live calls executed.')


if __name__ == '__main__':
    main()
