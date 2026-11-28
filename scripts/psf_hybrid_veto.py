#!/usr/bin/env python
"""Veto hybrids whose ePSF wing is net-negative beyond the blend radius (2026-11-28 recheck).

For those kernels the pure injected model (psf_kernel_model.fits) is the honest product:
the ePSF wing beyond wing_radius reads net-negative (a crowded/shallow-field floor, or the
backstop having zeroed a speckle-heavy wing asymmetrically), so the shipped kernel carries a
negative wing moat -- unphysical. The trimmed modelling kernel for ACS sits INSIDE the blend
radius and is identical to the model already, so the veto changes the archival kernel and
display only; it never changes a modelling result.

Per vetoed kernel:
    psf_kernel.fits      <- psf_kernel_model.fits (with its own inject_* header)
    stamps psf.fits      <- re-trimmed from the model kernel (header: PSFMETH -> inject)
    info/provenance/lens_psf.json: method -> inject_*, wing keys removed, trim size updated
    psf_kernel_model.fits, psf_kernel_empirical.fits, _blend, _empirical_err kept as-built
    (evidence); psf_err.fits kept (its trimmed window is the model core, already right).

Dry run by default; --apply writes. Idempotent (a vetoed kernel is no longer hybrid).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

ws_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import info_json  # noqa: E402
from make_psf import trim_kernel_to_amplitude  # noqa: E402

PSF_JSON = os.path.join(ws_path, 'info', 'provenance', 'lens_psf.json')
WING_KEYS = ('wing_radius_px', 'wing_frac', 'wing_n_stars', 'wing_err_frac', 'model_method')


def veto_kernel(kernel_path, hdr, wr, wing, apply):
    d = os.path.dirname(kernel_path)
    rel = os.path.relpath(d, os.path.join(ws_path, 'data', 'psf'))
    sample, lens, filt = rel.split(os.sep)[0], rel.split(os.sep)[3], rel.split(os.sep)[4]
    model_path = os.path.join(d, 'psf_kernel_model.fits')
    out = dict(path=rel, wing=wing)
    if not os.path.exists(model_path):
        out['error'] = 'no model sidecar'
        return out
    with fits.open(model_path) as h:
        mk, mh = h[0].data.astype(float), h[0].header
    inject_method = mh.get('PSFMETH', hdr.get('PSFMODEL', 'inject_unknown'))
    sub, size = trim_kernel_to_amplitude(mk, 1e-3)
    out['new_method'] = inject_method
    out['trim_size'] = size
    if apply:
        fits.PrimaryHDU(data=mk.astype(np.float32), header=mh).writeto(kernel_path,
                                                                       overwrite=True)
        stamp_dir = os.path.join(ws_path, 'data', 'stamps', rel)
        psf_path = os.path.join(stamp_dir, 'psf.fits')
        if os.path.exists(psf_path):
            th = fits.getheader(psf_path)
            th['PSFMETH'] = inject_method
            for stale in ('PSFWINGR', 'PSFWINGF', 'PSFMODEL'):
                if stale in th:
                    del th[stale]
            fits.PrimaryHDU(data=sub.astype(np.float32), header=th).writeto(psf_path,
                                                                            overwrite=True)
        # keep the trimmed error map shape-matched to the new kernel (2026-11-28: the old
        # hybrid's err crop can be a different size, leaving psf.fits/psf_err.fits mismatched)
        err_full = os.path.join(d, 'psf_kernel_err.fits')
        err_stamp = os.path.join(stamp_dir, 'psf_err.fits')
        if os.path.exists(err_full) and os.path.exists(err_stamp):
            e = fits.getdata(err_full).astype(float)
            cy = cx = e.shape[0] // 2
            half = (size - 1) // 2
            eh = fits.getheader(err_stamp)
            fits.PrimaryHDU(data=e[cy - half:cy + half + 1, cx - half:cx + half + 1]
                            .astype(np.float32), header=eh).writeto(err_stamp, overwrite=True)
        rec = (info_json.load(PSF_JSON).get(sample, {}).get(lens) or {}).get(filt)
        if rec:
            rec['method'] = inject_method
            rec['trimmed_kernel_size'] = int(size)
            rec['kernel_size'] = int(mk.shape[0])
            for key in WING_KEYS:
                rec.pop(key, None)
            info_json.update(PSF_JSON, sample, lens, filt, rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write the changes (dry run otherwise)')
    a = ap.parse_args()
    kernels = sorted(glob.glob(os.path.join(ws_path, 'data', 'psf', '*', '*', '*', '*', '*',
                                            'psf_kernel.fits')))
    rows = []
    for kp in kernels:
        hdr = fits.getheader(kp)
        m, wr = hdr.get('PSFMETH', ''), hdr.get('PSFWINGR')
        if not (str(m).startswith('hybrid') and wr is not None):
            continue
        k = fits.getdata(kp).astype(float)
        ny, nx = k.shape
        cy = cx = nx // 2
        yy, xx = np.mgrid[:ny, :nx]
        r = np.hypot(xx - cx, yy - cy)
        wing = float(k[r > float(wr)].sum())
        if wing < 0:
            rows.append(veto_kernel(kp, hdr, float(wr), wing, a.apply))
    print(f'{len(rows)} hybrids vetoed ({"applied" if a.apply else "dry run"}):')
    for r in sorted(rows, key=lambda x: x['wing']):
        print(f"  wing={r['wing']:+.3f} -> {r.get('new_method')}  {r['path']}"
              + (f"  [ERROR {r['error']}]" if 'error' in r else ''))
    log = os.path.join(ws_path, 'diagnostics', 'psf_recheck', 'hybrid_veto.json')
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'w') as f:
        json.dump(dict(apply=bool(a.apply), rows=rows), f, indent=1)
    print(f'wrote {log}')


if __name__ == '__main__':
    main()
