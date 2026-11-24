"""Every ``mammoth_*`` command in setup.py imports and answers ``--help``, and the docs only name
registered commands.

The commands are read from setup.py, so this runs without installing the package. It calls each
entry point's ``main()`` the way the console script would, in a fresh interpreter with the user
site-packages disabled (a stale ``~/.local`` transformers broke the HF converter on LUMI).
"""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _console_scripts():
    tree = ast.parse((REPO_ROOT / 'setup.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == 'console_scripts':
                    entries = [ast.literal_eval(elt) for elt in value.elts]
                    return dict(entry.replace(' ', '').split('=', 1) for entry in entries)
    raise AssertionError('no console_scripts in setup.py')


CONSOLE_SCRIPTS = _console_scripts()


@pytest.mark.parametrize('name', sorted(CONSOLE_SCRIPTS))
def test_help(name):
    module, func = CONSOLE_SCRIPTS[name].split(':')
    code = f'import sys; sys.argv = [{name!r}, "--help"]; from {module} import {func}; {func}()'
    env = {**os.environ, 'PYTHONNOUSERSITE': '1'}
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'usage' in result.stdout.lower(), result.stdout[-2000:]


def test_docs_name_only_registered_commands():
    # Skip file and directory names such as mammoth_hub.py or mammoth_pytorch/.
    command = re.compile(r'\bmammoth_[a-z0-9_]+(?![\w./])')
    docs = [REPO_ROOT / 'README.md', *(REPO_ROOT / 'docs' / 'source').rglob('*.md'),
            *(REPO_ROOT / 'docs' / 'source').rglob('*.rst')]
    unknown = {
        f'{path.relative_to(REPO_ROOT)}: {match}'
        for path in docs
        for match in command.findall(path.read_text())
        if match not in CONSOLE_SCRIPTS
    }
    assert not unknown, sorted(unknown)
