"""Inter-lead QRS, T-peak and T-offset delay matrices of every pre-processed ECG.

Reads every file <ecg-dir>/<eid>.npy ((12, 600) float32, R-peak at sample 0,
output of precompute_npy.py) and writes one npz with
  QRS, T_peak, T_offset : (K, 12, 12) float64, delay in ms
  filename              : (K,) str, '<eid>.npy', in directory-listing order
where delay[i, j] = (peak_j - peak_i) / fs * 1000, wrapped circularly to
[-300, 300) samples, so a positive value means lead i peaks before lead j.
A missing fiducial gives NaN in its row and column.

Per lead: R = argmax |5-15 Hz band-passed signal| in the first 150 ms of the
NeuroKit2-cleaned beat. T-peak and T-offset come from nk.ecg_delineate
(method 'dwt') with that R, keeping the first detection after R and at least
50 ms before the end of the window. A missing T-peak falls back to argmax
|1-7 Hz band-passed uncleaned beat| in [R + 80 ms, R + 400 ms), and a missing
T-offset to the first sample in [T-peak, T-peak + 200 ms) where that filtered
signal drops below 10% of its maximum absolute value in the range.

The output is dataset.delay_mat_path of src/configs/config_ecg_heart_delay.yaml
(delay_key T_peak) and config_ecg_heart_delay_qrs.yaml (delay_key QRS), and the
input of preliminary_ecg.ipynb (Fig. 1b).
"""
import argparse
import os
from pathlib import Path

import numpy as np
from scipy import signal as sp_signal
from tqdm import tqdm

try:
    import neurokit2 as nk
except ImportError:
    nk = None

REPO = Path(__file__).resolve().parents[1]


def wrap_to_cycle(delay_samples, num_samples):
    """Wrap signed sample differences into [-N/2, +N/2]."""
    return (delay_samples + num_samples // 2) % num_samples - num_samples // 2


def delay_matrix_from_peaks(peak_samples, fs, num_samples, circular=True):
    """
    Antisymmetric delay matrix (ms). delay[i, j] > 0 => lead i earlier than j.
    NaN peak propagates to NaN matrix entries.
    """
    peak_samples = np.asarray(peak_samples, dtype=float)
    delay_samples = np.subtract.outer(peak_samples, peak_samples) * -1
    if circular:
        finite = np.isfinite(delay_samples)
        wrapped = delay_samples.copy()
        wrapped[finite] = wrap_to_cycle(
            delay_samples[finite].astype(int), num_samples).astype(float)
        delay_samples = wrapped
    return delay_samples / fs * 1000.0


def find_fiducials_single_cycle(ecg_lead, fs, qrs_search_end_ms=150):
    """
    Fiducial detection for a single-cycle ECG segment that BEGINS with QRS.
    Constrains R-peak search to the first qrs_search_end_ms of the segment.
    """
    import warnings
    from scipy import signal as sp_signal

    result = {"R": np.nan, "T_peak": np.nan, "T_offset": np.nan}

    try:
        cleaned = nk.ecg_clean(ecg_lead, sampling_rate=fs)
        n = len(cleaned)

        # Restrict QRS search to the first ~150 ms (well within one QRS width)
        qrs_search_end = int(qrs_search_end_ms / 1000 * fs)

        # Polarity-invariant: find the sample of largest absolute amplitude
        # in the QRS search window, using QRS-band filtering
        sos = sp_signal.butter(3, (5, 15), btype='bandpass', fs=fs, output='sos')
        qrs_band = sp_signal.sosfiltfilt(sos, cleaned)
        r_peak = int(np.argmax(np.abs(qrs_band[:qrs_search_end])))
        result["R"] = r_peak

        # Delineate T-wave
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, waves = nk.ecg_delineate(
                cleaned, rpeaks=[r_peak], sampling_rate=fs, method="dwt")

        t_peaks_arr = np.asarray(waves.get("ECG_T_Peaks", []), dtype=float)
        t_peaks_arr = t_peaks_arr[np.isfinite(t_peaks_arr)]
        # T-peak must come AFTER R and BEFORE the end of the current cycle
        # (which is basically the whole segment since there's no next cycle in view)
        valid_t = t_peaks_arr[(t_peaks_arr > r_peak) & (t_peaks_arr < n - int(0.05*fs))]
        if len(valid_t) > 0:
            # Take the first T-peak after R (the one belonging to this beat)
            result["T_peak"] = int(valid_t[0])

        t_offs_arr = np.asarray(waves.get("ECG_T_Offsets", []), dtype=float)
        t_offs_arr = t_offs_arr[np.isfinite(t_offs_arr)]
        valid_toff = t_offs_arr[(t_offs_arr > r_peak) & (t_offs_arr < n - int(0.05*fs))]
        if len(valid_toff) > 0:
            result["T_offset"] = int(valid_toff[0])

    except Exception:
        pass

    return result


def fallback_t_peak(ecg_lead, fs, r_peak,
                    search_start_ms=80, search_end_ms=400, band=(1, 7)):
    """Argmax of |low-frequency signal| in [R+80ms, R+400ms]."""
    if not np.isfinite(r_peak):
        return np.nan
    sos = sp_signal.butter(3, band, btype='bandpass', fs=fs, output='sos')
    filtered = sp_signal.sosfiltfilt(sos, ecg_lead)
    n = len(ecg_lead)
    lo = int(r_peak) + int(search_start_ms / 1000 * fs)
    hi = min(n, int(r_peak) + int(search_end_ms / 1000 * fs))
    if hi <= lo:
        return np.nan
    return lo + int(np.argmax(np.abs(filtered[lo:hi])))


def fallback_t_offset(ecg_lead, fs, t_peak,
                      search_ms=200, band=(1, 7)):
    """
    Estimate T-offset as the point after the T-peak where the filtered signal
    returns closest to zero (simple tangent-to-baseline alternative).
    Looks in [T_peak, T_peak + search_ms].
    """
    if not np.isfinite(t_peak):
        return np.nan
    sos = sp_signal.butter(3, band, btype='bandpass', fs=fs, output='sos')
    filtered = sp_signal.sosfiltfilt(sos, ecg_lead)
    n = len(ecg_lead)
    lo = int(t_peak)
    hi = min(n, int(t_peak) + int(search_ms / 1000 * fs))
    if hi <= lo + 2:
        return np.nan
    # Offset ≈ where |filtered| drops below 10% of its peak absolute amplitude
    segment = np.abs(filtered[lo:hi])
    threshold = 0.1 * segment.max()
    below = np.where(segment < threshold)[0]
    if len(below) == 0:
        return np.nan
    return lo + below[0]


def all_delay_matrices_nk(data, fs, circular=True, lead_names=None,
                          use_fallback=True):
    """
    Compute three inter-lead delay matrices from multi-lead single-cycle ECG:
      1. QRS delay     (R-peak alignment)
      2. T-peak delay  (repolarization peak)
      3. T-offset delay (end of repolarization, ≈ electrical end-systole)

    Parameters
    ----------
    data : 2D array (num_leads, num_samples)
    fs : float
    circular : bool
        Wrap delays into [-cycle_ms/2, +cycle_ms/2].
    lead_names : list of str, optional
    use_fallback : bool
        If True, recover failed T-peak / T-offset detections with a
        band-filter-based heuristic.

    Returns
    -------
    matrices : dict of 2D arrays
        Keys: 'QRS', 'T_peak', 'T_offset' — each an antisymmetric ms matrix.
    fiducials : dict of 1D arrays
        Keys: 'R', 'T_peak', 'T_offset' — per-lead sample indices (NaN if failed).
    report : dict
        Per-wave counts of successful vs. failed detections.
    """
    num_leads, num_samples = data.shape
    if lead_names is None:
        lead_names = [f"Lead_{i}" for i in range(num_leads)]

    R       = np.full(num_leads, np.nan)
    Tpeak   = np.full(num_leads, np.nan)
    Toffset = np.full(num_leads, np.nan)

    for c in range(num_leads):
        fid = find_fiducials_single_cycle(data[c], fs)
        R[c]       = fid["R"]
        Tpeak[c]   = fid["T_peak"]
        Toffset[c] = fid["T_offset"]

        if use_fallback:
            if np.isnan(Tpeak[c]):
                Tpeak[c] = fallback_t_peak(data[c], fs, R[c])
            if np.isnan(Toffset[c]):
                # Prefer using detected T-peak if available
                ref_t = Tpeak[c]
                Toffset[c] = fallback_t_offset(data[c], fs, ref_t)

    matrices = {
        "QRS":      delay_matrix_from_peaks(R,       fs, num_samples, circular),
        "T_peak":   delay_matrix_from_peaks(Tpeak,   fs, num_samples, circular),
        "T_offset": delay_matrix_from_peaks(Toffset, fs, num_samples, circular),
    }
    fiducials = {"R": R, "T_peak": Tpeak, "T_offset": Toffset}
    report = {k: {"found": int(np.sum(np.isfinite(v))),
                  "missing": int(np.sum(~np.isfinite(v)))}
              for k, v in fiducials.items()}

    return matrices, fiducials, report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ecg-dir', default=str(REPO / 'data' / 'ukb_heartmesh_processed' / 'ecg_resting'),
                    help='Folder of <eid>.npy ECG files (dataset.ecg_dir).')
    ap.add_argument('--out', default=str(REPO / 'ukb_heart_mesh_visual' / 'all_ecg_delay_matrix.npz'),
                    help='Output npz (dataset.delay_mat_path of the Delay-T / Delay-QRS configs).')
    ap.add_argument('--fs', type=int, default=500, help='ECG sampling rate (Hz).')
    args = ap.parse_args()
    if nk is None:
        raise SystemExit('neurokit2 is required (pip install neurokit2==0.2.13)')

    ecg_flist = os.listdir(args.ecg_dir)
    all_ecg_delay_mat = {}
    for fn in tqdm(ecg_flist):
        ecg = np.load(os.path.join(args.ecg_dir, fn))
        delay_matrix = all_delay_matrices_nk(ecg, args.fs)[0]
        for k in delay_matrix:
            all_ecg_delay_mat.setdefault(k, []).append(delay_matrix[k])

    for k in all_ecg_delay_mat:
        all_ecg_delay_mat[k] = np.stack(all_ecg_delay_mat[k])
    all_ecg_delay_mat['filename'] = ecg_flist
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **all_ecg_delay_mat)
    print(f'Saved {len(ecg_flist)} delay matrices to {args.out}')


if __name__ == '__main__':
    main()
