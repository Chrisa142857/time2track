"""Pin the four bindings of a run: code, data, config, env.

  experiments/YYYY.MM.DD-runNN_slug/
      config.yaml  git_commit.txt  inputs.txt  env.txt  logs/  artifacts/


  python3 tools/runmeta.py new SLUG --key EID --question "..."   # create today's next run dir
  python3 tools/runmeta.py inputs RUN PATH [PATH ...]              # append inputs + hashes
  python3 tools/runmeta.py config RUN FILE.yaml                    # copy a config as config.yaml
  python3 tools/runmeta.py backfill                                # reconstruct for past runs

From Python, an experiment records the values it actually resolved with
    import runmeta; runmeta.write_config(run_dir, vars(args))

RUN is a run id, a key from experiments/INDEX.tsv, or a path. The interpreter whose
packages go into env.txt is T2T_PYTHON (from .env via paths.py).
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

RUN_RE = re.compile(r'^(\d{4})\.(\d{2})\.(\d{2})-run(\d{2})_[a-z0-9][a-z0-9.-]*$')
SKIP = {'__pycache__', 'Medformer', 'moment', 'EgoEMG', 'tbvenv', 'pylibs', '.git'}   # third-party trees


# ------------------------------------------------------------------------ helpers ----
def run_dir(ref):
    table = paths.runs()
    if ref in table:
        return table[ref]
    p = Path(ref)
    p = p if p.is_absolute() else (paths.EXPERIMENTS / ref if (paths.EXPERIMENTS / ref).is_dir() else p.resolve())
    if not p.is_dir() or not RUN_RE.match(p.name):
        sys.exit(f'not a run directory: {ref}')
    return p


def _yaml(value, indent=0):
    """Minimal YAML emitter: nested mappings, JSON flow style for every leaf."""
    pad = '  ' * indent
    out = []
    for k, v in value.items():
        key = k if re.fullmatch(r'[A-Za-z_][\w.-]*', str(k)) else json.dumps(str(k))
        if isinstance(v, dict) and v:
            out.append(f'{pad}{key}:')
            out.append(_yaml(v, indent + 1))
        else:
            out.append(f'{pad}{key}: {json.dumps(v, default=str)}')
    return '\n'.join(out)


def write_config(run, values, header=''):
    """Write the RESOLVED values a run used to <run>/config.yaml (never overwrites)."""
    f = Path(run) / 'config.yaml'
    if f.exists():
        raise FileExistsError(f'{f} exists: a run records its config once')
    f.write_text((header + '\n' if header else '') + _yaml(values) + '\n')
    return f


def git_record(when=None):
    git = ['git', '-C', str(paths.REPO)]
    sha = subprocess.run(git + ['rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if when:                                        # the commit that was HEAD on that date
        sha = subprocess.run(git + ['log', '-1', '--format=%h', f'--before={when} 23:59:59'],
                             capture_output=True, text=True).stdout.strip() or sha
    subject = subprocess.run(git + ['log', '-1', '--format=%ad %s', '--date=short', sha],
                             capture_output=True, text=True).stdout.strip()
    status = subprocess.run(git + ['status', '--porcelain'], capture_output=True, text=True).stdout
    return sha, subject, status


def env_record():
    py = str(paths.PYTHON)
    freeze = subprocess.run(['uv', 'pip', 'freeze', '--python', py], capture_output=True, text=True)
    if freeze.returncode:
        freeze = subprocess.run([py, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    probe = subprocess.run([py, '-c', 'import sys, torch; print(sys.version.split()[0], '
                            'torch.__version__, torch.version.cuda)'], capture_output=True, text=True)
    driver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,name', '--format=csv,noheader'],
                            capture_output=True, text=True).stdout.split('\n')[0]
    pyver, torchver, cuda = (probe.stdout.split() + ['?'] * 3)[:3]
    head = (f'python: {py}\npython_version: {pyver}\ntorch: {torchver}\ntorch_cuda: {cuda}\n'
            f'nvidia_driver_gpu: {driver or "none"}\n\n# uv pip freeze\n')
    return head + freeze.stdout


def _sha_file(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(bs), b''):
            h.update(b)
    return h.hexdigest()


def hash_input(p, max_bytes, max_files):
    """-> (hash, size note). Strongest to weakest, each labelled:
    sha256            file content
    tree-sha256       content of every file below a directory
    listing-sha256    relpath/size/mtime of every file (directory above max_bytes)
    top-listing-sha256  size/mtime of the immediate entries (walk exceeded max_files)
    top-names-sha256  names of the immediate entries only, one directory read (more than
                      max_files entries: stat-ing each over a shared NFS mount is not worth it)
    stat-sha256       size/mtime of a file above max_bytes"""
    p = Path(p)
    if p.is_file():
        st = p.stat()
        if st.st_size <= max_bytes:
            return f'sha256:{_sha_file(p)}', f'{st.st_size} bytes'
        h = hashlib.sha256(f'{st.st_size}\t{st.st_mtime_ns}'.encode()).hexdigest()
        return f'stat-sha256:{h}', f'{st.st_size} bytes, size and mtime only'
    top = sorted(os.listdir(p))
    if len(top) > max_files:
        h = hashlib.sha256('\n'.join(top).encode()).hexdigest()
        return f'top-names-sha256:{h}', f'{len(top)} entries, names only'
    files, capped = [], False
    for root, dirs, names in os.walk(p, followlinks=True):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for n in sorted(names):
            q = Path(root) / n
            try:
                st = q.stat()
            except OSError:
                continue                              # dangling link
            files.append((str(q.relative_to(p)), st.st_size, st.st_mtime_ns, q))
        if len(files) > max_files:
            capped = True
            break
    h = hashlib.sha256()
    if capped:
        for n in top:
            st = (p / n).stat()
            h.update(f'{n}\t{st.st_size}\t{st.st_mtime_ns}\n'.encode())
        return f'top-listing-sha256:{h.hexdigest()}', f'{len(top)} entries, more than {max_files} files below'
    total = sum(f[1] for f in files)
    if total <= max_bytes:
        for rel, size, _, q in files:
            h.update(f'{rel}\t{size}\t{_sha_file(q)}\n'.encode())
        kind = 'tree-sha256'
    else:
        for rel, size, mtime, _ in files:
            h.update(f'{rel}\t{size}\t{mtime}\n'.encode())
        kind = 'listing-sha256'
    return f'{kind}:{h.hexdigest()}', f'{len(files)} files, {total} bytes'


def append_inputs(run, items, max_bytes, max_files, memo=None):
    memo = {} if memo is None else memo
    f = Path(run) / 'inputs.txt'
    new = not f.exists()
    with open(f, 'a') as out:
        if new:
            out.write('# path\thash\tsize\n# sha256 / tree-sha256 hash content. The weaker labels are explained in '
                      'tools/runmeta.py:hash_input (listing, top-listing, top-names, stat).\n')
        for p in items:
            key = os.path.realpath(p)
            if key not in memo:
                memo[key] = hash_input(p, max_bytes, max_files)
            out.write(f'{p}\t{memo[key][0]}\t{memo[key][1]}\n')
            out.flush()


# ----------------------------------------------------------------------- commands ----
def cmd_new(a):
    today = datetime.date.today().strftime('%Y.%m.%d')
    taken = [int(RUN_RE.match(d.name)[4]) for d in paths.EXPERIMENTS.iterdir()
             if d.name.startswith(today + '-run') and RUN_RE.match(d.name)]
    rid = f'{today}-run{max(taken, default=0) + 1:02d}_{a.slug}'
    if not RUN_RE.match(rid):
        sys.exit(f'bad slug {a.slug!r}: lower-case letters, digits, "-" and "." only')
    if a.key in paths.runs():
        sys.exit(f'key {a.key} already in {paths.INDEX}')
    d = paths.EXPERIMENTS / rid
    (d / 'logs').mkdir(parents=True)
    (d / 'artifacts').mkdir()
    sha, subject, status = git_record()
    (d / 'git_commit.txt').write_text(f'commit: {sha}  ({subject})\n'
                                      f'tree: {"DIRTY, not reproducible" if status else "clean"}\n'
                                      f'git status --porcelain:\n{status}')
    (d / 'env.txt').write_text(env_record())
    with open(paths.INDEX, 'a') as f:
        f.write(f'{a.key}\t{rid}\t{a.question}\n')
    print(d)


def cmd_inputs(a):
    append_inputs(run_dir(a.run), a.paths, a.max_gb * 2**30, a.max_files)


def cmd_config(a):
    d = run_dir(a.run)
    if (d / 'config.yaml').exists():
        sys.exit(f'{d}/config.yaml exists: a run records its config once')
    (d / 'config.yaml').write_text(Path(a.file).read_text())


# ---------------------------------------------------------------------- backfill -----
RECON = 'RECONSTRUCTED 2026-09-28 during the repository reorganization; not recorded at run time.'


def _literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.JoinedStr):              # f'{paths.X}/rest' -> resolved path
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append(str(v.value))
            elif (isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Attribute)
                  and isinstance(v.value.value, ast.Name) and v.value.value.id in ('paths', 'repo_paths')):
                parts.append(str(getattr(paths, v.value.attr)))
            else:
                return None
        return ''.join(parts)
    return None


SEED_CALL = re.compile(r'(^|\.)(seed|manual_seed|manual_seed_all|default_rng|RandomState|set_seed|seed_everything)$')


def scan_code(f):
    """constants, argparse defaults and literal seeds of one script."""
    tree = ast.parse(Path(f).read_text(errors='replace'))
    consts, args, seeds = {}, {}, []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                names = [t] if isinstance(t, ast.Name) else list(getattr(t, 'elts', []))
                vals = [node.value] if isinstance(t, ast.Name) else list(getattr(node.value, 'elts', []))
                for n, v in zip(names, vals):
                    if isinstance(n, ast.Name) and n.id.isupper():
                        lit = _literal(v)
                        try:
                            short = lit is not None and len(json.dumps(lit, default=str)) < 400
                        except TypeError:             # e.g. a dict keyed by tuples: record it as text
                            lit, short = repr(lit), len(repr(lit)) < 400
                        if short:
                            consts[n.id] = lit
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = ast.unparse(node.func)
        if fn.endswith('add_argument') and node.args and isinstance(node.args[0], ast.Constant):
            kw = {k.arg: k.value for k in node.keywords}
            flag = str(node.args[-1].value if isinstance(node.args[-1], ast.Constant) else node.args[0].value)
            if 'default' in kw:
                args[flag] = _literal(kw['default'])
            elif isinstance(kw.get('action'), ast.Constant) and kw['action'].value == 'store_true':
                args[flag] = False
            if 'seed' in flag.lower() and args.get(flag) is not None:
                seeds.append(f'{flag} default {args[flag]}')
        elif SEED_CALL.search(fn) and node.args:
            lit = _literal(node.args[0])
            if isinstance(lit, int):
                seeds.append(f'{fn}({lit})')
        for k in node.keywords:
            if k.arg and 'seed' in k.arg.lower() and isinstance(_literal(k.value), int):
                seeds.append(f'{fn}(..., {k.arg}={_literal(k.value)})')
    for name, v in consts.items():
        if 'SEED' in name and isinstance(v, (int, list, tuple)):
            seeds.append(f'{name} = {v}')
    return consts, args, seeds


def run_code(d):
    for p in sorted(d.rglob('*.py')):
        rel = p.relative_to(d)
        if set(rel.parts) & SKIP or any('venv' in x for x in rel.parts):
            continue
        if rel.parts[0] == 'artifacts' and 'src' in rel.parts[1:]:
            continue                                   # training-time code snapshot
        yield p


def external_inputs(d, code):
    """Locations the run's code reads outside its own directory (static paths.* strings,
    or the static directory prefix of a dynamic one)."""
    found = set()
    for f in code:
        try:
            tree = ast.parse(f.read_text(errors='replace'))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            parts, dyn, uses = [], False, False
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif (isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Attribute)
                      and isinstance(v.value.value, ast.Name) and v.value.value.id in ('paths', 'repo_paths')):
                    parts.append(str(getattr(paths, v.value.attr)))
                    uses = True
                else:
                    dyn = True
                    break
            if not uses:
                continue
            s = ''.join(parts)
            s = os.path.realpath(s.rsplit('/', 1)[0] if dyn else s)
            p = Path(s)
            if p == d or d in p.parents or p in (paths.REPO, paths.REPO / 'src') or not os.path.lexists(s):
                continue
            if p.suffix in ('.py', '.sh') or 'python' in p.name:
                continue                              # code and interpreters are pinned by git/env
            found.add(s)
    for cfg in sorted(d.glob('configs/*.y*ml')):
        for m in re.finditer(r'^\s*\w*(?:_dir|_path|_root)\s*:\s*([^\s#]+)', cfg.read_text(), re.M):
            v = m[1].strip('"\'')
            s = os.path.realpath(v if v.startswith('/') else paths.REPO / 'src' / v)
            p = Path(s)
            if v and os.path.lexists(s) and not (p == d or d in p.parents):
                found.add(s)
    # a directory already covered by a listed parent is not listed twice
    return sorted(s for s in found if not any(o != s and s.startswith(o + '/') for o in found))


def cmd_backfill(a):
    env = env_record()
    memo = {}
    for key, d in paths.runs().items():
        m = RUN_RE.match(d.name)
        when = f'{m[1]}-{m[2]}-{m[3]}'
        code = list(run_code(d))
        if not (d / 'git_commit.txt').exists():
            sha, subject, status = git_record(when)
            (d / 'git_commit.txt').write_text(
                f'# {RECON}\ncommit: {sha}  (HEAD on the run date {when}: {subject})\n'
                'tree: DIRTY at run time. The run used uncommitted edits to src/ and untracked\n'
                'analysis code in this run directory, and the exact tree was not recorded, so this\n'
                'run must not be quoted as reproducible (versioning rule 3).\n'
                f'git status --porcelain at reconstruction:\n{status}')
        if not (d / 'env.txt').exists():
            (d / 'env.txt').write_text(f'# {RECON}\n# Captured on 2026-09-28, not at run time: packages '
                                       'may have changed since the run.\n' + env)
        if not (d / 'config.yaml').exists():
            scripts, seeds = {}, []
            for f in code:
                try:
                    consts, args, s = scan_code(f)
                except SyntaxError:
                    continue
                entry = {k: v for k, v in (('constants', consts), ('argparse_defaults', args)) if v}
                if entry:
                    scripts[str(f.relative_to(d))] = entry
                seeds += [f'{f.relative_to(d)}: {x}' for x in s]
            cfgs = sorted(str(p.relative_to(d)) for p in d.glob('configs/*.y*ml'))
            values = {'run_id': d.name, 'key': key, 'date': when,
                      'seed': sorted(set(seeds)) or 'no literal seed in this run\'s code '
                      '(unseeded or deterministic; unverified)'}
            if cfgs:
                values['training_configs'] = cfgs
            values['scripts'] = scripts
            write_config(d, values, header=f'# {RECON}\n# Values are the literals in the run\'s code '
                         '(module constants, argparse defaults, seeds). A value overridden on the\n'
                         '# command line at run time is not recoverable here unless the run\'s logs record it.')
        if not (d / 'inputs.txt').exists():
            items = external_inputs(d, code)
            append_inputs(d, items, a.max_gb * 2**30, a.max_files, memo)
            with open(d / 'inputs.txt') as f:
                body = f.read()
            (d / 'inputs.txt').write_text(f'# {RECON}\n# Inputs are the locations the run\'s code reads outside '
                                          'this directory, hashed on 2026-09-28.\n' + body)
        print(f'{key:10s} {d.name}  ({len(code)} scripts)', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('new')
    p.add_argument('slug')
    p.add_argument('--key', required=True, help='short id used by paths.py, e.g. E29')
    p.add_argument('--question', required=True, help='the question or hypothesis, one line')
    p = sub.add_parser('inputs')
    p.add_argument('run')
    p.add_argument('paths', nargs='+')
    p.add_argument('--max-gb', type=float, default=float(os.environ.get('T2T_HASH_MAX_GB', 2)),
                   help='directories above this size get a listing hash instead of a content hash')
    p.add_argument('--max-files', type=int, default=int(os.environ.get('T2T_HASH_MAX_FILES', 20000)),
                   help='directories with more files than this get a top-level fingerprint')
    p = sub.add_parser('config')
    p.add_argument('run')
    p.add_argument('file')
    p = sub.add_parser('backfill')
    p.add_argument('--max-gb', type=float, default=float(os.environ.get('T2T_HASH_MAX_GB', 2)))
    p.add_argument('--max-files', type=int, default=int(os.environ.get('T2T_HASH_MAX_FILES', 20000)))
    a = ap.parse_args()
    {'new': cmd_new, 'inputs': cmd_inputs, 'config': cmd_config, 'backfill': cmd_backfill}[a.cmd](a)


if __name__ == '__main__':
    main()
