"""TARE: the two-stage apparent-scale removal pipeline of the paper.

Module map, against Section III:

    config      Table III, plus the DIPSER layout settings
    dataset     III-E, sampling and the target (1)
    landmarks   III-B1 and III-C2, the MediaPipe extraction, z_i and u_i
    geometry    (2) and the references of III-B2
    features    III-C4, the RAW, TARE-S and TARE-FULL sets, (4) to (6)
    models      III-C5, the regressors, the folds and B1 to B4 of (7)
    audit       III-C6, ERROR, GAP and LEAK, (8) to (16)
    zoom        III-C7 and III-C8, the perturbation protocols, (17) to (21)
    stats       III-C9, the bootstrap, the sign-flip test and Holm
"""

from __future__ import annotations

__all__ = [
    "audit", "config", "dataset", "features", "geometry",
    "landmarks", "models", "stats", "zoom",
]
