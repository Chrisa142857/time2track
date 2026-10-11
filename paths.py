"""Single source of truth for every filesystem location in this project.

Nothing else in the repository may spell out an absolute path. Three kinds of location:

  * Inside the repository: derived from this file's own location (REPO, EXPERIMENTS, ...).
  * Machine roots: read from the environment or from REPO/.env, never restated here.
    A missing root fails loudly instead of falling back to a guess. The .env file is
    machine-specific and untracked (template: .env.example).
  * Run directories: resolved through experiments/INDEX.tsv, so a run can be renamed
    in one place. `paths.E16` is the run directory of experiment E16 and
    `paths.E16_ws` is its verbatim original workspace, experiments/<run>/artifacts.

Python:  import paths;  paths.PYTHON;  f'{paths.E16_ws}/npy'
Bash:    eval "$(python3 <repo>/paths.py --sh)";  "$T2T_PYTHON" script.py
Check:   python3 paths.py        (prints every location and verifies it exists)
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXPERIMENTS = REPO / 'experiments'
SUPERSEDED = REPO / 'superseded'
DOCS = REPO / 'docs'
SUBMISSION = REPO.parent / 'neurips26_submission'
INDEX = EXPERIMENTS / 'INDEX.tsv'

_ROOTS = {'RAM': 'T2T_RAM_ROOT', 'WORK': 'T2T_WORK_ROOT', 'CNS': 'T2T_CNS_ROOT',
          'PYTHON': 'T2T_PYTHON'}


def _dotenv():
    env = {}
    f = REPO / '.env'
    if f.is_file():
        for line in f.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                v = v.split(' #', 1)[0].split('\t#', 1)[0]     # trailing comment
                env[k.strip()] = v.strip().strip('"\'')
    return env


def _root(name):
    var = _ROOTS[name]
    value = os.environ.get(var) or _dotenv().get(var)
    if not value:
        raise RuntimeError(f'{var} is not set: define it in {REPO / ".env"} '
                           f'(template: .env.example) or export it')
    return Path(value)


def runs():
    """{key: run directory} from experiments/INDEX.tsv (empty in a clone without experiments/)."""
    if not INDEX.is_file():
        return {}
    rows = [l.split('\t') for l in INDEX.read_text().splitlines()[1:] if l.strip()]
    return {r[0]: EXPERIMENTS / r[1] for r in rows}


def __getattr__(name):                       # PEP 562: resolved on first use
    if name in _ROOTS:
        return _root(name)
    table = runs()
    if name in table:
        return table[name]
    if name.endswith('_ws') and name[:-3] in table:
        return table[name[:-3]] / 'artifacts'
    raise AttributeError(f'paths has no location {name!r} (not a root, not in {INDEX})')


def _all():
    out = {'REPO': REPO, 'EXPERIMENTS': EXPERIMENTS, 'SUPERSEDED': SUPERSEDED,
           'DOCS': DOCS, 'SUBMISSION': SUBMISSION}
    out.update({k: _root(k) for k in _ROOTS})
    for k, d in runs().items():
        out[k] = d
        out[f'{k}_ws'] = d / 'artifacts'
    return out


if __name__ == '__main__':
    table = _all()
    if sys.argv[1:] == ['--sh']:
        for k, v in table.items():
            print(f'export T2T_{k}={str(v)!r}')
        sys.exit(0)
    missing = []
    for k, v in table.items():
        ok = v.exists() or k.endswith('_ws')          # not every run had a workspace
        print(f'{"ok " if ok else "!! "} {k:14s} {v}')
        if not ok:
            missing.append(k)
    assert not missing, f'missing locations: {missing}'
    print(f'\n{len(table)} locations resolved')
