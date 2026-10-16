"""Write one MRI-aligned resting ECG CSV per (mesh, ECG file) pair.

Inputs (UK Biobank bulk layout, one level of sub-folders below each root):
  --mesh-root  <sub>/<eid>_30099_<instance>_0.zip    cardiac MR meshes (only the names are read)
  --ecg-root   <sub>/<eid>_20205_<instance>_0.xml    resting 12-lead ECG
  --hr-csv     CSV indexed by `eid` with columns `12673-<instance>.<array>`

For every participant with both a mesh and a resting ECG, and for every mesh
of that participant, the heart rate of the mesh instance is the mean of the
non-missing `12673-<instance>.*` values. A mesh without a heart rate is
skipped. Each ECG file of the participant is then aligned as follows.

  1. The 600-sample, 500 Hz median beat (`MedianSamples` of the XML) is read
     and scaled by its `Resolution` to microvolts.
  2. On every lead, nk.ecg_clean + nk.ecg_peaks (NeuroKit2, method 'neurokit')
     detect R-peaks and the first one is kept. The median over leads is the
     consensus R-peak (argmax of the lead with the largest range if no lead
     yields a peak).
  3. The beat is rolled circularly so that the consensus R-peak is sample 0,
     the start of the ECG-gated MR cycle (frame 0 of the mesh).

The shifted (12, 600) signals are written to <out-dir>/<eid>_<instance>_<k>.csv
(rows = leads in the XML order, columns = samples), where k enumerates the
participant's ECG files in directory-listing order. The stretched time axis
60 / heart rate is computed but not written. precompute_npy.py turns these
CSVs into dataset.ecg_dir.
"""
import argparse
import os
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import neurokit2 as nk
except ImportError:
    nk = None

warnings.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[1] / 'data' / 'ukb_heartmesh_processed'


def list_bulk(root, ext):
    """{participant id: [paths]} for files <root>/<sub>/<name><ext>, in listing order."""
    out = {}
    for d in os.listdir(root):
        for f in os.listdir(os.path.join(root, d)):
            if f.endswith(ext):
                out.setdefault(f.split('_')[0], []).append(os.path.join(root, d, f))
    return out


def get_ecg_signal(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the MedianSamples section
    median_samples = root.find('.//MedianSamples')
    if median_samples is None:
        raise ValueError("MedianSamples not found in XML")

    # Get resolution
    resolution = float(median_samples.find('Resolution').text)

    signals = []
    leads = []
    # Iterate over WaveformData elements
    for waveform in median_samples.findall('WaveformData'):
        lead = waveform.get('lead')
        leads.append(lead)
        data_str = waveform.text.strip()
        # Split by comma and convert to float
        data = np.array([float(x) for x in data_str.split(',')])
        # Convert to microvolts
        data *= resolution
        signals.append(data)

    return np.stack(signals), leads


def get_ecg_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract ECG Heart Rate
    hr_node = root.find('.//VentricularRate')
    if hr_node is None:
        raise ValueError("VentricularRate not found in XML.")
    hr_ecg = float(hr_node.text)

    # Extract Sample Rate
    sample_rate_node = root.find('.//SampleRate')
    sample_rate = float(sample_rate_node.text) if sample_rate_node is not None else 500.0

    return hr_ecg, sample_rate


def get_cmr_hr(subj_cmrhr, eid_value, insid):
    # Heart rate of the imaging instance, field 12673
    tgt_header = [h for h in subj_cmrhr.columns if h.split('.')[0] == f'12673-{insid}']
    tgt_val = subj_cmrhr.loc[int(eid_value), tgt_header]
    return tgt_val[~tgt_val.isna()].mean()


def consensus_r_peak(signals, sr):
    """Consensus R-peak index of a (n_leads, n_samples) beat: NeuroKit2 ecg_clean + ecg_peaks
    ('neurokit') on every lead, first R-peak per lead, median across leads. Falls back to the
    argmax of the lead with the largest amplitude range when no lead yields a peak."""
    r_peak_votes = []
    for i in range(signals.shape[0]):
        try:
            lead_clean = nk.ecg_clean(signals[i].astype(float), sampling_rate=sr, method='neurokit')
            _, info = nk.ecg_peaks(lead_clean, sampling_rate=sr, method='neurokit')
            r_peaks = info['ECG_R_Peaks']
            if len(r_peaks) > 0:
                r_peak_votes.append(int(r_peaks[0]))
        except Exception:
            continue

    if len(r_peak_votes) > 0:
        return int(np.median(r_peak_votes))
    # Fallback: argmax on the lead with the largest amplitude range
    lead_idx = int(np.argmax(signals.max(axis=1) - signals.min(axis=1)))
    return int(np.argmax(signals[lead_idx]))


def process_and_map_ecg_to_mri(xml_path, mri_hr):
    """
    Shift ECG data and return a stretched time axis to temporally align with
    the heart mesh (CMR). The signal itself is NOT resampled.

    Shift: runs nk.ecg_clean + nk.ecg_peaks on every lead independently,
           collects the first detected R-peak index from each lead, then
           takes the median across leads as the consensus R-peak (multi-lead
           voting). The signals are rolled so the consensus R-peak lands at
           index 0. Falls back to amplitude-based selection if nk fails on
           all leads.

    Stretch (time axis only): the returned time_axis maps the n_samples ECG
           samples onto one CMR cardiac cycle duration (60 / mri_hr seconds),
           correcting for any heart-rate discrepancy between modalities without
           altering the signal values.

    Returns
    -------
    dict with keys
        ecg_signals  : np.ndarray (n_leads, n_samples)  – original resolution
        ecg_leads    : list[str]
        time_axis    : np.ndarray (n_samples,)  – seconds, spans 60 / mri_hr
        hr_ecg       : float  – ECG heart rate (bpm)
        mri_hr       : float  – CMR heart rate (bpm)
        sample_rate  : float  – ECG sample rate (Hz)
        r_peak_idx   : int    – consensus R-peak sample index (before shift)
    """
    signals, leads = get_ecg_signal(xml_path)       # (n_leads, n_samples)
    hr_ecg, sample_rate = get_ecg_metadata(xml_path)

    # Shift: multi-lead voting for R-peak
    r_peak_idx = consensus_r_peak(signals, int(sample_rate))
    signals = np.roll(signals, -r_peak_idx, axis=1)

    # Stretch via time axis: map n_samples onto one CMR cardiac cycle
    #   duration = 60 / mri_hr  seconds  (no signal resampling)
    n_samples = signals.shape[1]
    time_axis = np.linspace(0, 60.0 / mri_hr, n_samples, endpoint=False)

    return {
        'ecg_signals': signals,
        'ecg_leads':   leads,
        'time_axis':   time_axis,
        'hr_ecg':      hr_ecg,
        'mri_hr':      mri_hr,
        'sample_rate': sample_rate,
        'r_peak_idx':  r_peak_idx,
    }


def _save_ecg_for_mesh(xml_paths, zip_path, out_root, subj_cmrhr):
    """
    Given a single mesh zip and a list of ECG xml paths for the same subject,
    compute mri_hr, run process_and_map_ecg_to_mri for each ECG file, and
    save results as CSVs.

    Output filename: {eid}_{insid}_{ecg_idx}.csv
    Rows = ECG leads, columns = aligned time samples.
    Returns number of files written.
    """
    zip_fname = os.path.basename(zip_path)
    eid   = int(zip_fname.split('_')[0])
    insid = int(zip_fname.split('_')[-2])

    mri_hr = get_cmr_hr(subj_cmrhr, eid, insid)
    if np.isnan(mri_hr):
        return 0

    written = 0
    for ecg_idx, xml_path in enumerate(xml_paths):
        out_path = os.path.join(out_root, f'{eid}_{insid}_{ecg_idx}.csv')
        if os.path.exists(out_path):          # skip already-done files
            written += 1
            continue
        try:
            data = process_and_map_ecg_to_mri(xml_path, mri_hr)
        except Exception:
            continue
        df_out = pd.DataFrame(data['ecg_signals'], index=data['ecg_leads'])
        df_out.to_csv(out_path)
        written += 1

    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mesh-root', required=True, help='Root of the mesh zip files (<sub>/<eid>_30099_*.zip).')
    ap.add_argument('--ecg-root', required=True, help='Root of the resting ECG XML files (<sub>/<eid>_20205_*.xml).')
    ap.add_argument('--hr-csv', required=True, help='Heart-rate table: index column eid, columns 12673-<instance>.<array>.')
    ap.add_argument('--out-dir', default=str(DATA / 'ecg_resting_csv'), help='Output folder for the aligned CSVs.')
    args = ap.parse_args()
    if nk is None:
        raise SystemExit('neurokit2 is required (pip install neurokit2==0.2.13)')

    mesh = list_bulk(args.mesh_root, '.zip')
    ecgrest = list_bulk(args.ecg_root, '.xml')
    subj_cmrhr = pd.read_csv(args.hr_csv, index_col='eid')
    os.makedirs(args.out_dir, exist_ok=True)

    print("Saving MRI-aligned resting ECGs …")
    n_rest, err_rest = 0, 0
    for subj in tqdm(sorted(set(mesh) & set(ecgrest))):
        for zip_path in mesh[subj]:
            try:
                n_rest += _save_ecg_for_mesh(ecgrest[subj], zip_path, args.out_dir, subj_cmrhr)
            except Exception:
                err_rest += 1
    print(f"Resting done: {n_rest} files saved, {err_rest} errors.")


if __name__ == '__main__':
    main()
