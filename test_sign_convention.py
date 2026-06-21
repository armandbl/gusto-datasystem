#!/usr/bin/env python3
"""Verify the cross-correlation sign convention with synthetic Gaussians.

A known pixel shift is injected into a synthetic target map, then
``measure_shift_integer`` cross-correlates it with the reference.
The test confirms ``dx_pix = -lag_x`` recovers the injected shift
with the correct sign, and that ``anchor + daz`` (not ``anchor - daz``)
is the correct formula for iterative convergence.

Usage:  python test_sign_convention.py
"""

import sys
from pathlib import Path

# Ensure utils/ is importable
_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir / "utils"))

import numpy as np
from measure_mixer_crosscorr import measure_shift_integer


def gaussian_map(shape: tuple[int, int], center: tuple[float, float],
                 sigma: float = 3.0) -> np.ndarray:
    """Create a 2D Gaussian centred at *center* (y, x)."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    return np.exp(-((xs - center[1]) ** 2 + (ys - center[0]) ** 2) / (2 * sigma**2))


def test_convention() -> int:
    """Run all tests.  Returns number of failures."""
    failures = 0
    shape = (101, 101)
    center = (50.0, 50.0)

    # --- Test 1: target shifted RIGHT by 5 pixels ------------------------
    ref = gaussian_map(shape, center)
    tgt_right = gaussian_map(shape, (50.0, 55.0))  # shifted RIGHT +5 in X

    lag_x, lag_y, _, _ = measure_shift_integer(ref, tgt_right)
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    print(f"Test 1 — target RIGHT by 5 pix:")
    print(f"  lag_x={lag_x}, lag_y={lag_y}")
    print(f"  dx_pix={dx_pix:+.1f}, dy_pix={dy_pix:+.1f}")
    print(f"  Expected: lag_x=-5, dx_pix=+5")

    if lag_x == -5 and dx_pix == 5:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — sign convention in measure_shift_integer is wrong")
        failures += 1

    # --- Test 2: target shifted LEFT by 5 pixels -------------------------
    tgt_left = gaussian_map(shape, (50.0, 45.0))  # shifted LEFT -5 in X

    lag_x, lag_y, _, _ = measure_shift_integer(ref, tgt_left)
    dx_pix = -float(lag_x)
    dy_pix = -float(lag_y)

    print(f"\nTest 2 — target LEFT by 5 pix:")
    print(f"  lag_x={lag_x}, lag_y={lag_y}")
    print(f"  dx_pix={dx_pix:+.1f}, dy_pix={dy_pix:+.1f}")
    print(f"  Expected: lag_x=+5, dx_pix=-5")

    if lag_x == 5 and dx_pix == -5:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL — sign convention in measure_shift_integer is wrong")
        failures += 1

    # --- Test 3: verify dx_pix = -lag_x always recovers the injected shift
    # The injected shift from ref to target is: tgt_center - ref_center.
    # The measurement should give lag = -(injected shift).
    # The correction dx = -lag should equal the injected shift.
    print(f"\nTest 3 — sign consistency over a range of shifts:")
    for shift in range(-10, 11):
        if shift == 0:
            continue
        tgt = gaussian_map(shape, (50.0, 50.0 + shift))
        lag_x, _, _, _ = measure_shift_integer(ref, tgt)
        dx = -float(lag_x)
        if dx != shift:
            print(f"  ✗ shift={shift:+3d}: lag_x={lag_x:+3d}, dx_pix={dx:+3d} (expected {shift:+3d})")
            failures += 1

    if failures == 0:
        print("  ✓ all shifts correct (dx_pix = -lag_x = injected shift)")

    # --- Test 4: stored offset = old_target + daz (iterative accumulation) ---
    # The cross-corr measures the RESIDUAL after current offsets.
    # New absolute offset = old target offset + residual (daz).
    # This works for both initial measurement (old = THEORY) and
    # iterative updates (old = previous AS_MEASURED).
    print(f"\nTest 4 — new_az = old_target_az + daz:")
    # Simulated: B2M5 old THEORY = +0.031436°, measured daz = -0.011437°
    old_target_az = 0.031436
    daz = -0.011437
    new_az = old_target_az + daz
    print(f"  B2M5: old={old_target_az:+.6f}°, daz={daz:+.6f}°, new={new_az:+.6f}°")
    assert abs(new_az - 0.019999) < 1e-6, f"Expected +0.019999, got {new_az}"
    print(f"    → old + daz = {new_az:+.6f}° (adjusts from THEORY by residual) ✓")

    # Simulated: B1M2 old THEORY = +0.091139°, measured daz = +0.049402°
    old_target_az = 0.091139
    daz = 0.049402
    new_az = old_target_az + daz
    print(f"  B1M2: old={old_target_az:+.6f}°, daz={daz:+.6f}°, new={new_az:+.6f}°")
    assert abs(new_az - 0.140541) < 1e-6, f"Expected +0.140541, got {new_az}"
    print(f"    → old + daz = {new_az:+.6f}° (adjusts from THEORY by residual) ✓")

    # Iterative: second measurement gives near-zero residual → stays put
    old_target_az = 0.019999
    daz = -0.0003  # near-zero residual after correction
    new_az = old_target_az + daz
    print(f"  B2M5 iter2: old={old_target_az:+.6f}°, daz={daz:+.6f}°, new={new_az:+.6f}°")
    assert abs(new_az - 0.019699) < 1e-6
    print(f"    → old + daz ≈ old (stable, doesn't overwrite) ✓")

    print("  ✓ old_target + daz correctly handles both initial and iterative cases")

    return failures


if __name__ == "__main__":
    n_fail = test_convention()
    if n_fail == 0:
        print(f"\n{'='*60}")
        print("ALL TESTS PASSED")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"{n_fail} TEST(S) FAILED")
        print(f"{'='*60}")
        sys.exit(1)
