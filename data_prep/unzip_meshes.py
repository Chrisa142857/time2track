"""Unzip the UK Biobank cardiac MR mesh bulk files (<eid>_30099_2_0.zip).

Every archive under --src (searched recursively) is extracted to
--dst/<relative path without .zip>/, so the sub-folder layout of the bulk
download is kept. An archive whose target folder already holds the
*pos_file.txt and *shift_file.txt members is skipped.
"""
import argparse
import zipfile
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm


def unzip_file(zip_path, src_root, dest_root):
    """Unzip one archive to dest_root, preserving its path relative to src_root."""
    try:
        # subfolder/file.zip -> subfolder/file/
        target_folder = dest_root / zip_path.relative_to(src_root).with_suffix('')
        if any(target_folder.glob('*pos_file.txt')) and any(target_folder.glob('*shift_file.txt')):
            return f'Existed {target_folder}'
        target_folder.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_folder)
        return f'Successfully unzipped {zip_path}'
    except Exception as e:
        return f'Failed to unzip {zip_path}: {e}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', type=Path, required=True, help='Root of the downloaded mesh zip files.')
    ap.add_argument('--dst', type=Path, required=True, help='Root for the unzipped subject folders.')
    ap.add_argument('--num-workers', type=int, default=4, help='Parallel unzip processes.')
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    zip_files = list(args.src.rglob('*.zip'))
    if not zip_files:
        print('No zip files found in the source directory or its subfolders.')
        return
    print(f'Found {len(zip_files)} zip files. Unzipping with {args.num_workers} processes...')
    worker = partial(unzip_file, src_root=args.src, dest_root=args.dst)
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(worker, zip_files), total=len(zip_files),
                            desc='Unzipping files'))
    for result in results:
        print(result)


if __name__ == '__main__':
    main()
