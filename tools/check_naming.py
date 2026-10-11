"""Machine check of the project's naming and run-layout rules. Token lists and the
grandfather date live in tools/naming.ini.

  python3 tools/check_naming.py            # full audit of the tree (warnings + errors)
  python3 tools/check_naming.py --staged   # pre-commit: staged names + experiments/ layout

Errors (exit 1):
  * a name with a bare qualifier token (final, old, best, ...) and no metric or seed token
  * an experiments/ directory not named YYYY.MM.DD-runNN_slug, or missing from INDEX.tsv
  * a run without config.yaml, git_commit.txt, inputs.txt, env.txt
  * a run dated on/after `since` without logs/ and artifacts/, or with a top-level
    artifact whose name carries no metric or seed
  * a superseded/ entry that its folder README does not name together with its replacement
Grandfathered runs (dated before `since`) report bare names as warnings only.
"""
import argparse
import configparser
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

RUN_RE = re.compile(r'^(\d{4})\.(\d{2})\.(\d{2})-run(\d{2})_[a-z0-9][a-z0-9.-]*$')
METRIC = re.compile(r'^[A-Za-z][A-Za-z-]*\d*\.\d+([-,]\d*\.?\d+)*$')    # sep0.196-0.219, val-loss0.0123
BINDINGS = ('config.yaml', 'git_commit.txt', 'inputs.txt', 'env.txt')
TOP_FILES = re.compile(r'^(INDEX\.tsv|README\.md|MIGRATION_.*)$')
DOC_EXT = {'.py', '.sh', '.md', '.log', '.yaml', '.yml', '.txt', '.ipynb'}


def load_rules(f):
    c = configparser.ConfigParser()
    if not c.read(f):
        sys.exit(f'naming rules not found: {f}')
    s = c['naming']
    return {k: set(s[k].split()) for k in ('bare', 'split', 'exempt', 'skip')} | {'since': s['since']}


def stem(name):
    return re.sub(r'(\.[A-Za-z][A-Za-z0-9]{0,5})+$', '', name)   # drop extensions, keep decimals


def carries_metric(name):
    return any(METRIC.match(t) for t in re.split(r'_', stem(name))) or bool(re.search(r'seed\d+', name))


def bare_tokens(name, rules):
    toks = [t.lower() for t in re.split(r'[_\-.\s]+', name) if t]
    bad = [t for t in toks if t in rules['bare'] and t not in rules['exempt']]
    return [] if not bad or carries_metric(name) else bad


def check_names(names, rules, report, level='error'):
    for p in names:
        bad = bare_tokens(os.path.basename(p), rules)
        if bad:
            report(level, p, f'bare qualifier {"/".join(bad)} with no metric or seed in the name')


def walk_names(root, rules, depth):
    base = len(str(root).rstrip('/').split('/'))
    for r, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in rules['skip'] and 'venv' not in d]
        if len(r.split('/')) - base >= depth:
            dirs[:] = []
        for n in dirs + files:
            yield os.path.join(r, n)


def check_experiments(rules, report, deep):
    if not paths.EXPERIMENTS.is_dir():
        return                                    # a clone without the private experiments/
    index = paths.runs()
    listed = {d.name for d in index.values()}
    for e in sorted(paths.EXPERIMENTS.iterdir()):
        if e.name in rules['skip']:
            continue                              # caches, e.g. __pycache__ from an import
        rel = e.relative_to(paths.REPO)
        if e.is_file():
            if not TOP_FILES.match(e.name):
                report('error', rel, 'stray file at experiments/ top level (belongs in a run)')
            continue
        m = RUN_RE.match(e.name)
        if not m:
            report('error', rel, 'run directory not named YYYY.MM.DD-runNN_slug')
            continue
        if e.name not in listed:
            report('error', rel, 'run directory missing from experiments/INDEX.tsv')
        for f in BINDINGS:
            if not (e / f).exists():
                report('error', rel / f, 'missing run binding')
        new = f'{m[1]}.{m[2]}.{m[3]}' >= rules['since']
        if new:
            for d in ('logs', 'artifacts'):
                if not (e / d).is_dir():
                    report('error', rel / d, 'missing run subdirectory')
            art = e / 'artifacts'
            for a in sorted(art.iterdir()) if art.is_dir() else []:
                toks = {t.lower() for t in re.split(r'[_\-.]+', a.name)}
                if a.is_file() and a.suffix not in DOC_EXT and not carries_metric(a.name) \
                        and not toks & rules['split']:
                    report('error', a.relative_to(paths.REPO), 'artifact name carries no metric or seed')
        if deep or new:
            check_names([os.path.relpath(p, paths.REPO) for p in walk_names(e, rules, 3)], rules,
                        report, 'error' if new else 'warn')
    for d in index.values():
        if not d.is_dir():
            report('error', d.relative_to(paths.REPO), 'listed in INDEX.tsv but does not exist')


def check_superseded(rules, report):
    if not paths.SUPERSEDED.is_dir():
        return
    listed = {d.name for d in paths.runs().values()}
    for d in sorted(paths.SUPERSEDED.iterdir()):
        rel = d.relative_to(paths.REPO)
        if d.name not in listed and d.name != 'docs':
            report('error', rel, 'superseded/ folder must be named after a run id (or docs)')
        readme = d / 'README.md'
        text = readme.read_text() if readme.is_file() else ''
        if not text:
            report('error', rel / 'README.md', 'superseded folder without README naming replacements')
        lines = text.splitlines()
        for f in sorted(d.iterdir()):
            if f.name == 'README.md':
                continue
            if not any(f'`{f.name}`' in l and 'replac' in l.lower() for l in lines):
                report('error', f.relative_to(paths.REPO), 'README does not name this file with its replacement')


def self_test(rules):
    flagged = ['model_final_v2_updated.pth', 'edges_v8.jsonl', 'results_old.json', 'e26.py.bak_pooled',
               'best_model', 'es_OLD_rawgt.npz', 'diag_treat_final.json']
    passing = ['best_val-loss0.012345_step900', 'edges_g1g2g3_sep0.196-0.219-0.217_seed42.jsonl',
               'ac_drop_V2', 'fig1b_A_v3.npz', 'val_index.json', 'newton.py', 'final_seed7.npz']
    wrong = [n for n in flagged if not bare_tokens(n, rules)] + [n for n in passing if bare_tokens(n, rules)]
    assert not wrong, f'self-test failed on {wrong}'
    print('self-test ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--staged', action='store_true', help='check staged files only (pre-commit)')
    ap.add_argument('--self-test', action='store_true', help='check the rules against known names')
    ap.add_argument('--rules', default=os.environ.get('T2T_NAMING_RULES',
                                                      str(Path(__file__).with_name('naming.ini'))))
    a = ap.parse_args()
    rules = load_rules(a.rules)
    if a.self_test:
        return self_test(rules)
    found = {'error': [], 'warn': []}

    def report(level, path, msg):
        found[level].append(f'{path}: {msg}')

    if a.staged:
        staged = subprocess.run(['git', '-C', str(paths.REPO), 'diff', '--cached', '--name-only',
                                 '--diff-filter=ACR'], capture_output=True, text=True).stdout.split()
        check_names([p for p in staged if not set(Path(p).parts) & rules['skip']], rules, report)
    else:
        tracked = subprocess.run(['git', '-C', str(paths.REPO), 'ls-files'],
                                 capture_output=True, text=True).stdout.split()
        check_names([p for p in tracked if not set(Path(p).parts) & rules['skip']], rules, report)
    check_experiments(rules, report, deep=not a.staged)
    check_superseded(rules, report)

    for w in found['warn']:
        print('warn ', w)
    for e in found['error']:
        print('ERROR', e)
    print(f'{len(found["error"])} errors, {len(found["warn"])} warnings (grandfathered: runs before '
          f'{rules["since"]})', file=sys.stderr)
    sys.exit(1 if found['error'] else 0)


if __name__ == '__main__':
    main()
