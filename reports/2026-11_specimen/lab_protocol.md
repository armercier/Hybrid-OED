# Lab protocol: setup day before the weekend LDV scan

Plate: 500 x 250 x 1 mm metal sheet (light, presumably aluminium), four 1 g / 10 mm PZT discs, LDV scan
on the front face, defects machined from the back face.

**Goal of the day:** plate characterised, acquisition parameters fixed, a pristine reference scan
(before machining) saved, defects machined, and the dense scan running unattended over the weekend.

**Order matters:** acquisition parameters are fixed *before* the pristine scan, so that the pristine
scan and the weekend scan are directly comparable. Machining comes *after* the pristine scan.

---

## Coordinate convention (decide once, write it on the plate)

- Origin: bottom-left corner seen from the **front (LDV) face**; x along the 500 mm side, y along the
  250 mm side. This is the frame of the Salvus simulation (`src/salvus_specimen/build_and_run.py`, plate `thin`).
- PZT discs, pits and the glued cube are on the **back face**. Seen from the back, x is mirrored:
  **x_back = 500 - x**. Through-holes are symmetric; pits and the cube are not.

| Item | Front-face position (mm) | Back-face x (mm) | Notes |
|---|---|---|---|
| S1 (PZT) | (250, 20) | 250 | |
| S2 (PZT) | (420, 125) | 80 | bonded last (scattering test) |
| S3 (PZT) | (250, 230) | 250 | |
| S4 (PZT) | (80, 125) | 420 | |
| D1 | (195, 160) | 305 | through-hole, diameter 12 mm |
| D2 | (295, 170) | 205 | pit, diameter 20 mm, 0.5 mm deep, back face |
| D3 | (250, 95) | 250 | through-hole, diameter 6 mm |
| D4 | (185, 85) | 315 | pit, diameter 10 mm, 0.5 mm deep, back face |
| M1 | (310, 95) | 190 | glued steel cube, 5 mm (about 1 g), back face |
| Scan area | x 150 to 350, y 45 to 205 | | 200 x 160 mm, front face |

---

## Schedule

### 08:00-09:00 Plate characterisation and preparation

- [ ] Measure the real dimensions (ruler).
- [ ] Thickness with a micrometer at >= 9 points (3 x 3 grid + scan centre); note mean and spread.
      A0 speed scales with sqrt(h): 5 % thickness error = 2.5 % speed error.
- [ ] Weigh the plate. Density = mass / (L x W x h): about 2.7 g/cm3 = aluminium, about 7.9 g/cm3 = steel.
- [ ] Support: lay the plate freely on 4-6 small soft foam pads near corners/edges, away from the
      scan area. No clamps.
- [ ] Pencil marks: source positions, defect positions (front and back face, mirrored), scan corners.
- [ ] Three small fiducial marks visible to the LDV camera, outside the scan area (re-registration if
      the plate moves during machining).
- [ ] LDV surface: thin uniform developer / whitening spray over the scan area. Bare aluminium is
      specular (poor signal off-centre). Avoid patches of retro-reflective tape (on a 1 mm plate
      their edges scatter); if tape is unavoidable, cover an area larger than the scan, uniformly.
- [ ] Weigh the four PZT discs and the steel cube; note each mass.

### 09:00-10:00 Piezo tests and bonding

- [ ] Electrical check before bonding: capacitance with an LCR meter, all four within about +-10 %.
      Reject outliers.
- [ ] Bond **S1, S3, S4** on the back face. Thin stiff bond (cyanoacrylate or thin fast epoxy), same
      glue and similar layer for all. Respect the cure time. **Leave S2 unbonded for now.**
- [ ] Point check per source: fire each bonded disc, LDV at a point about 30 mm from it (scan side).
      Same drive -> peak amplitudes within about +-3 dB and similar waveforms. Rebond if one is much
      weaker or ringing.
- [ ] Electrical crosstalk: look at the LDV trace at t = 0. A spike at the trigger is electrical
      pickup; reroute / ground cables until it is gone or well before the first arrival.

### 10:00-10:45 Do the PZT discs scatter? (one before/after test)

- [ ] With **S4 firing**, scan a 60 x 60 mm patch around the future S2 position at 2 mm (about 900
      points), provisional settings (about 30 averages).
- [ ] Bond **S2**; after curing, repeat the identical patch scan.
- [ ] Difference (after - before) = scattering of one 1 g disc. Compare with the noise:
  - above noise -> the inversion must include the discs as known masses (the simulation already
    does): record their masses and positions precisely;
  - at noise level -> discs can be ignored.
- [ ] Point check S2 like the others.

### 10:45-12:00 Acquisition parameters (on the undrilled plate)

1. **Excitation:** 40 kHz, Ricker-like pulse or 3-5 cycle Hann-windowed burst. Same waveform and
   voltage for all four sources. Record the generator/amplifier output on a reference channel in
   every scan.
2. **Linearity:** at one point, record at about half and full voltage; the traces must scale exactly.
   If not, lower the voltage.
3. **Sampling:** >= 625 kHz (about 16 x f0).
4. **Record length:** >= 1.5 ms after the trigger + about 0.1 ms pre-trigger (used for the noise estimate).
5. **Repetition rate (ring-down), critical on an undamped metal plate:** fire once and record about
   50 ms; find when the ring-down falls below the noise. The repetition period must be longer than
   that, otherwise the tail of the previous shot is coherent with the trigger, does not average out
   and corrupts the data. Expect a few ms to tens of ms.
6. **Averages vs SNR:** at the worst point (scan corner farthest from the weakest source), record
   with 10, 30, 100, 300 averages. SNR = peak / noise rms (noise from the pre-trigger), should grow
   as sqrt(N). **Target: SNR >= 90 at the worst point** (the level assumed in all simulations).
   Choose the smallest N that reaches it.
7. **Signal-level map:** quick coarse scan (10 mm, few averages) of the whole area to check focus and
   signal level everywhere. Fix dropouts (surface, focus, angle) now.
8. **Time per point:** T_point = N x T_rep + scan overhead (measure the overhead on the coarse scan).

### 12:00-14:00 Pristine (undrilled) reference scan, runs over lunch

- [ ] All four sources, final parameters, grid chosen to fit about 2 h in total:
      points per source = 7200 / (4 x T_point).
      Example: T_point = 2 s -> about 900 points per source -> 6 mm grid (34 x 27 = 918) or 8 mm (26 x 21 = 546).
- Why: on the real plate this calibrates thickness, material, source wavelets and the effect of the
  discs, and it is the measured pristine baseline for the inversion.

### 14:00-15:30 Machining

- [ ] Machine in place if possible (plate on its support, same position). If it must be moved,
      re-register it with the fiducials afterwards.
- [ ] Protect the bonded discs and their cables; do not press on the plate near them.
- [ ] Through-holes **D1 (12 mm)** and **D3 (6 mm)**: drill on a backing block to avoid bending the
      sheet; deburr.
- [ ] Pits **D2 (20 mm)** and **D4 (10 mm)** from the **back face (mirrored x)**: sand / grind gently
      to about 0.5 mm deep. Measure the real depth (dial gauge or micrometer at the centre) and
      diameter; photograph with a ruler; note the profile if possible (it will not be flat).
- [ ] Glue the steel cube **M1** on the back face (mirrored x), same glue as the discs; note its mass.
- [ ] Remove all chips and dust (loose chips add mass and noise).

### 15:30-16:15 Post-machining checks

- [ ] Repeat the morning point check per source: same amplitude within about +-1 dB -> bonds survived.
- [ ] Quick line scan across D1 and M1 (2 mm step, final settings): scattered waves visible above noise.

### 16:15-17:30 Define and launch the dense scan

- [ ] Time available from launch to your return (e.g. Friday 17:30 to Monday 08:00 = 62 h);
      keep a 20 % margin -> about 50 h.
- [ ] Points per source = 50 h x 3600 / (4 x T_point). Choose the finest grid that fits:

| Grid over 200 x 160 mm | Points per source | T_point = 2 s | T_point = 4 s | T_point = 6 s |
|---|---|---|---|---|
| 2 mm (as simulated) | 8181 | 18 h, fits | 36 h, fits | 55 h, too long |
| 2.5 mm | 5265 | fits | 23 h, fits | 35 h, fits |
| 3 mm | 3685 | fits | fits | 25 h, fits |
| 4 mm (minimum; used by the inversion) | 2091 | fits | fits | fits |

(Hours = total for the four sources.)

- [ ] One complete scan per source, run in sequence (S1, S2, S3, S4), each saved as its own file.
      If something stops mid-weekend, the finished sources are complete datasets. If time is tight,
      run 4 mm for all four sources first, then refine.
- [ ] Drift check: if the software allows, a fixed reference point every N points; at least
      re-measure one point at the start and end of each source scan.
- [ ] Temperature logger next to the plate, if available.
- [ ] Before leaving: test-run the first ~20 points and check the files are saved; check disk space
      (points x samples x channels); disable PC sleep and automatic updates.

---

## Log sheet

- Plate: dimensions, thickness map, mass, material.
- PZT discs and cube: masses, exact positions (both faces), glue type.
- Defects: real diameters, depths, profiles, photos.
- Excitation: waveform, frequency, voltage, amplifier gain.
- Acquisition: sampling rate, record length, pre-trigger, repetition rate, averages, SNR at the worst point.
- Scans: grid, origin in plate coordinates, file name per source, start/end times.
- Temperature; anything unusual.

## Monday morning

- [ ] All four scans complete.
- [ ] Look at one wavefield snapshot per source.
- [ ] Compare the start/end reference-point traces (drift).
- [ ] Copy the data off the lab PC.
