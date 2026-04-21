"""Minimal local AutoSIMP helpers needed by the GPU-FEM workflow."""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import pub_simp_solver as solver


def install_passive_patch(passive_mask: Optional[np.ndarray]) -> None:
    """
    Patch the OC update so passive regions are enforced during bisection.
    """
    if not hasattr(solver, "_oc_update_original"):
        solver._oc_update_original = solver._oc_update

    if passive_mask is None:
        solver._oc_update = solver._oc_update_original
        return

    orig_oc = solver._oc_update_original
    mask = passive_mask
    void_idx = np.where(mask == 1)[0]
    solid_idx = np.where(mask == 2)[0]
    has_void = len(void_idx) > 0
    has_solid = len(solid_idx) > 0

    def _oc_update_with_passive(rho, dc, dv, volfrac, move, n_elem, beta, eta, use_heaviside, H_mat):
        dc = dc.copy()
        dv = dv.copy()
        if has_void:
            dc[void_idx] = -1e-20
            dv[void_idx] = 1e-20
        if has_solid:
            dc[solid_idx] = -1e-20
            dv[solid_idx] = 1e-20

        rho_new = orig_oc(rho, dc, dv, volfrac, move, n_elem, beta, eta, use_heaviside, H_mat)
        if has_void:
            rho_new[void_idx] = 1e-3
        if has_solid:
            rho_new[solid_idx] = 1.0
        return rho_new

    solver._oc_update = _oc_update_with_passive


def uninstall_passive_patch() -> None:
    install_passive_patch(None)
