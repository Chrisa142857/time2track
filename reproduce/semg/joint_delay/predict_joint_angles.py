"""Ground-truth and predicted joint angles for the Fig. 1d hand pose delay analysis.

Runs the released emg2pose regression_vemg2pose checkpoint on the start of each emg2pose session
(samples [start, stop) at the native 2 kHz) and writes one npz per session with
`gt` and `pred` (T x 20 joint angles) plus the session metadata (including `stage`).

Port of the inference cells of emg2pose's notebooks/getting_started.ipynb: same config overrides,
same checkpoint loading, same window, same skip rule. As in the notebook, the module stays on the device
the checkpoint is mapped to on load (the first visible GPU if there is one, otherwise CPU).
"""
import argparse
import glob
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--emg2pose-repo', default=os.environ.get('EMG2POSE_REPO'),
                    help='clone of github.com/facebookresearch/emg2pose @5f6f62b (env EMG2POSE_REPO)')
    ap.add_argument('--data-dir', required=True, help='directory with the emg2pose *.hdf5 session files')
    ap.add_argument('--checkpoint', required=True, help='emg2pose_model_checkpoints/regression_vemg2pose.ckpt')
    ap.add_argument('--out-dir', required=True, help='output directory for preds_and_gts_<i>.npz')
    ap.add_argument('--experiment', default='tracking_vemg2pose',
                    help='hydra experiment override, as in getting_started.ipynb (the module itself is '
                         'rebuilt from the hyperparameters stored in --checkpoint)')
    ap.add_argument('--n-sessions', type=int, default=444,
                    help='use the first N session files in sorted filename order')
    ap.add_argument('--start', type=int, default=0, help='window start sample (2 kHz)')
    ap.add_argument('--stop', type=int, default=10_000, help='window stop sample (2 kHz)')
    args = ap.parse_args()
    if not args.emg2pose_repo:
        ap.error('--emg2pose-repo (or env EMG2POSE_REPO) is required')

    repo = Path(args.emg2pose_repo).resolve()
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from tqdm import tqdm
    from emg2pose.data import Emg2PoseSessionData
    from emg2pose.lightning import Emg2PoseModule
    from emg2pose.utils import generate_hydra_config_from_overrides

    sessions = sorted(glob.glob(os.path.join(args.data_dir, '*.hdf5')))[:args.n_sessions]
    if len(sessions) < args.n_sessions:
        sys.exit(f'found {len(sessions)} hdf5 files in {args.data_dir}, expected at least {args.n_sessions}')
    data_list = [Emg2PoseSessionData(hdf5_path=s) for s in sessions]

    config = generate_hydra_config_from_overrides(
        config_path=str(repo / 'config'),
        overrides=[f'experiment={args.experiment}', f'checkpoint={args.checkpoint}'])
    module = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for i, session in enumerate(tqdm(data_list)):
        if session['emg'].shape[0] < module.model.left_context:
            continue
        session_window = session[args.start:args.stop]
        # no_ik_failure is not a field so we slice separately
        no_ik_failure_window = session.no_ik_failure[args.start:args.stop]
        batch = {
            'emg': torch.Tensor([session_window['emg'].T]).to(module.device),  # BCT
            'joint_angles': torch.Tensor([session_window['joint_angles'].T]).to(module.device),  # BCT
            'no_ik_failure': torch.Tensor([no_ik_failure_window]).to(module.device),  # BT
        }
        preds, joint_angles, no_ik_failure = module.forward(batch)
        if (joint_angles[:, 0] == 0).all():
            print('Warning! Ground truth not available at first time step!')
        np.savez(f'{args.out_dir}/preds_and_gts_{i}.npz',
                 gt=joint_angles[0].cpu().T.detach().numpy(),   # BCT -> TC
                 pred=preds[0].cpu().T.detach().numpy(),
                 **session.metadata)


if __name__ == '__main__':
    main()
