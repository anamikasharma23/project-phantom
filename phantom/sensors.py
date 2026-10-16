"""The public sensor model.

These formulas describe how WALL-E's instruments behave. They are "public
knowledge": both the environment (to generate readings) and the agent (to
interpret them) use the same formulas. What the agent does NOT know is
hidden truth such as EVE's position, her status, or how trustworthy each
signal source really is.
"""
import math

import numpy as np

STATUSES = ("active", "dormant")
STRENGTH_NOISE = 0.08          # std-dev of beacon strength readings
CLUE_TRACK_NOISE_DEG = 10.0    # std-dev of a physical track clue's bearing
CLUE_TRACK_RELIABILITY = 0.9   # documented spec of track clues
RESIDUE_ACCURACY = 0.85        # residue clue reads status / proximity correctly
RESIDUE_NEAR_RADIUS = 8.0      # "near" means EVE within this many cells


def kappa_from_noise(noise_deg):
    """Concentration of a von Mises bearing distribution from a std-dev."""
    sigma = math.radians(max(noise_deg, 1.0))
    return 1.0 / (sigma * sigma)


def von_mises_pdf(delta_deg, kappa):
    """Density (per radian) of a bearing error of `delta_deg`.

    Uses the scaled form exp(k(cos d - 1)) / (2*pi*i0e(k)) which stays
    numerically stable for large k.
    """
    d = np.radians(delta_deg)
    return np.exp(kappa * (np.cos(d) - 1.0)) / (2.0 * math.pi * np.i0(kappa) * math.exp(-kappa))


UNIFORM_BEARING_PDF = 1.0 / (2.0 * math.pi)


def beacon_hear_prob(dist, status, spec):
    """Chance that one long-range scan hears EVE's beacon."""
    emit = spec["emit_active"] if status == "active" else spec["emit_dormant"]
    return emit * np.exp(-np.asarray(dist) / spec["range"])


def beacon_strength_mean(dist, spec):
    return np.exp(-np.asarray(dist) / spec["range"])
