# TARE

An implementation of the methodology in *Apparent Scale Bias in
Landmark-Based Classroom Student Engagement Estimation: A Controlled,
Demographic-Unaware Fairness Audit*. Every quantity comes
from Section III and Table III of that paper. Nothing has been added to the
method and nothing has been left out of it.

## What this does and does not contain

No frame, landmark, rating or result is generated here. The code reads the
DIPSER release off disk and stops with an error when something it needs is
missing, so a missing dataset gives a failed run and never a filled-in one.
DIPSER frames are not redistributed, per the paper's data disclosure.

The unit tests in `tests/` check Proposition 1, Proposition 2 and Remark 3.
Those are identities that hold for any numbers, and Remark 3 calls the
keypoint-level case a unit test itself, so they are verified on arbitrary
arrays. No test produces a research result.

## Layout

    tare/config.py      Table III, and where the DIPSER files sit
    tare/dataset.py     III-E: 1.5 s sampling, the rating fill, the target (1)
    tare/landmarks.py   III-B1, III-C2: MediaPipe, z_i and u_i
    tare/geometry.py    (2) and the references of III-B2
    tare/features.py    III-C4: RAW, TARE-S, TARE-FULL, (4) to (6)
    tare/models.py      III-C5: regressors, subject-wise folds, B1 to B4
    tare/audit.py       III-C6: ERROR, GAP, LEAK, (8) to (16)
    tare/zoom.py        III-C7, III-C8: the perturbations, (17) to (21)
    tare/stats.py       III-C9: bootstrap, sign-flip, Holm, equivalence
    run_audit.py        Algorithm 1, line by line

## Running it

    python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt

Download `pose_landmarker.task` and `face_landmarker.task` from the
MediaPipe model index. Section III-B1 requires a fresh extraction, so the
pose annotations DIPSER ships are never read.

Inspect the labels before the first full run:

    ./.venv/bin/python run_audit.py --dipser-root PATH \
        --pose-task pose_landmarker.task --face-task face_landmarker.task \
        --inspect-labels

Check the layout before anything else, because the file patterns in
`DipserLayout` are settings rather than facts:

    ./.venv/bin/python run_audit.py --dipser-root PATH \
        --pose-task pose_landmarker.task --face-task face_landmarker.task \
        --describe-only

Then run the audit:

    ./.venv/bin/python run_audit.py --dipser-root PATH \
        --pose-task pose_landmarker.task --face-task face_landmarker.task \
        --session-order sessions.csv --regressor gb

`--regressor ridge` runs the other family of III-C5. Both use the Table III
settings for every variant, so a difference between variants comes from the
features. The extraction pass is cached, so a rerun of the audit does not
repeat it; the ZOOM branches re-extract from warped images and are not
cached, which is where the time goes.

## What you still have to supply

**The dataset.** It is not on this machine. DIPSER is CC BY 4.0 and is
requested through SciDB, with the reading scripts at
`https://bitbucket.org/rovitlib/dipser/`. Unzip it anywhere and pass the
directory as `--dipser-root`; nothing has to sit inside this project.

The release is organised as `group<N>/experiment<M>/subject<K>.zip`, and each
subject archive holds `images`, `watch_sensors`, `labels` and `metadata`.
The glob defaults in `tare/config.DipserLayout` follow that, with
`activity_pattern` controlling whether the directory reads `experiment3` or
`experiment03`. Unzip the subject archives before running, then check with
`--describe-only`.

DIPSER experiments 3, 5 and 8 are the lecture, the lecture test and the
robotics experimentation, which are the three activities III-E names.

**The recording order.** Stage H2 calibrates on the first 60 s of "the
earliest session in time", so `--session-order` needs a CSV with columns
`student`, `activity`, `start`. Frame numbers restart each session and
cannot supply this, and guessing it would change which frames form C_s, so
the loader refuses to run without it.

**Which DIPSER group is "group 01".** Assumption A6 names the 16 students of
group 01, and III-F says DIPSER records post-secondary students. In the
DIPSER release the 16-student group is the master's-degree group, while the
other two groups hold up to 20 students each and are early childhood
education. `config.DIPSER_GROUP` is set to `01` as the paper writes it, so
if the release numbers that master's group differently, change that one
value and say so in the paper.

**Which evaluator is the self-assessment.** DIPSER stores five evaluations
per student, four experts and one self-evaluation, and (1) averages the four
experts only. The loader reads four files and refuses a partial set rather
than averaging whatever it finds, so run `--inspect-labels` to see the five
and set `self_evaluation_key` before trusting the target.

**Confirmation of the four pending definitions.** Section III-B2 leaves the
face center, shoulder tilt, neck tilt and ear asymmetry as `[PENDING]`.
`tare/geometry.py` implements the anatomical descriptions you supplied, and
each function records which description it came from and why it is written
the way it is. Two constraints from the paper narrowed the choices: an angle
has to survive a scaling and a translation for Proposition 1 to hold, and
the face center has to be a function of the eleven keypoints for Remark 3 to
be the algebraic identity it claims to be. Read those four docstrings and
confirm they say what you meant.

## A number to reconcile

Table III gives the desk camera as 640x480 at about 8.7 frames per second,
and the DIPSER paper gives its individual cameras as 640x480 at about 10 fps.
`config.FRAME_RATE` holds 8.7, as Table III says, and `_frame_time` converts
a frame number to t_i with it. The rate only affects which frame falls in
which 1.5 s bin, so a wrong value shifts the sampling slightly rather than
breaking it. Check it against the real recordings and settle the paper and
the config on one figure.

## Three readings worth knowing about

These are places where the paper is silent and the code had to settle on
something. Each is marked in the source.

The median window alignment of (4) is a setting,
`config.MEDIAN_WINDOW_ALIGNMENT`, with centered as the default, which
matches Table II describing the window as sitting around frame i.

Frames with `u_i = 0` are left out of fitting and receive the training mean
as their prediction, per III-C5. Their features are missing throughout, so
there is nothing to fit on. Median imputation still applies to a frame that
is usable but has some component undefined.

Out-of-fold leakage predictions are averaged over the three seeds before
R-squared is computed in (15), following the way (8) averages over seeds
inside each student.

## Traceability

Every numbered item of Section III and where it lives.

| Paper | Where |
| --- | --- |
| (1) target, mean of four experts | `dataset._target_at` |
| (2) shoulder midpoint and width | `geometry.shoulder_midpoint`, `geometry.shoulder_width` |
| (3) weak-perspective pixel length | not implemented, see below |
| (4) in-frame scale L and its windowed median | `geometry.in_frame_scale`, `features.in_frame_scales` |
| (5) keypoints relative to m in units of L-bar | `features.tare_s_vector` |
| (6) stage H2 baseline and calibrated features | `features.student_baselines`, `features.tare_full_matrix` |
| (7) B1 to B4 | `models.reference_predictors`, `models.human_reference` |
| (8) frame and student error | `audit.frame_errors`, `audit.student_errors` |
| (9) ERROR and the gains over RAW | `audit.error_summary` |
| (10) largest student error and spread | `audit.error_summary` |
| (11) equalized-odds gap | `audit.equalized_odds` and the two forms around it |
| (12) change in the gap | `audit.gap_change` |
| (13) rank association and its change | `audit.rank_association`, `audit.association_change` |
| (14) the two apparent-scale proxies | `audit.apparent_scale_proxies` |
| (15) leakage R-squared | `audit.leakage` |
| (16) leakage Xi | `audit.leakage` |
| (17) zoom-and-shift map | `zoom.apply_keypoint_map`, `zoom.affine_matrix` |
| (18) drift | `zoom.drift` |
| (19) threshold-crossing rate | `zoom.flip_rate` |
| (20) dose-response curve and its slope | `zoom.dose_response`, `zoom.log_linear_fit` |
| (21) perturbed-cohort disparity | `zoom.perturbed_cohort` |
| Proposition 1, exact invariance | `tests/test_propositions.py` |
| Proposition 2, centering | `tests/test_propositions.py` |
| Remark 3, keypoint drift is a unit test | `tests/test_propositions.py` |
| Algorithm 1, lines 1 to 20 | `run_audit.py` |
| A1 to A7 | A1 and A2 in `geometry`/`features`, A3 in `landmarks` and `dataset`, A4 in `dataset`, A5 in `audit.apparent_scale_proxies`, A6 in `dataset.load_cohort`, A7 in `features.student_baselines` |

Intervals follow III-C9. The mean gain, the quantities of (10), the gaps of
(11), the association of (13), the leakage of (15) and (16), the drift of
(18), the slope of (20) and the spreads of (21) each come with a bootstrap
over students, with a student's group carried along. The primary family of
ten comparisons gets the sign-flip test on ERROR, the bootstrap share for
the gap changes, and Holm at 0.05.

## Not implemented, on purpose

Equation (3), the weak-perspective decomposition of a pixel length into
scale, true length and foreshortening, has no code. III-C3 says the model
"serves only to interpret the origin of scale" and that the invariance in
Proposition 1 does not rely on it, so implementing it would mean choosing a
focal length and a camera distance the paper never gives. Remark 1 reads the
five residual mechanisms off it, and those are read from the measurements
rather than computed.

## Not implemented, on purpose

III-E keeps the five-vote majority label as a secondary target and then
marks the secondary analysis as `[PENDING]`, noting that no measure in
Section III uses it. It is therefore not read. Once you define that
analysis, the label has to be loaded in `dataset.py`.

III-B1 reports a correlation of r = 0.99 on interocular distance and
r = 0.49 on shoulder width between a fresh MediaPipe run and the DIPSER
metadata, with the frame count, correlation type and the manual check of
shoulder width all marked `[PENDING]`. The code never mixes the two sources,
which is the conclusion the paper draws, so reproducing that comparison is a
separate script and not part of the pipeline.

III-G marks the MediaPipe and scikit-learn versions, the hardware and the
code availability statement as `[PENDING]`. `requirements.txt` has what this
was tested against.
