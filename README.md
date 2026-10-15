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

## What the real files settled

The release was inspected before any of this was wired up, and four things
Section III leaves open are now decided by the data rather than by a guess.

**Which group.** Assumption A6 names the 16 students of "group 01". In the
release, group_01 has 19 students with all three activities and group_03 has
15, while **group_02 has exactly 16**, which is also the only group for which
"all of them have a lecture session, so N_lec = 16" holds. `DIPSER_GROUP` is
therefore `02`, and A6 should be corrected in the paper.

**Which evaluator to drop.** Each session has five label files: four named
`labeler_*` and one named `self_labeling`. `rating_glob` matches only the
first kind, so the self-assessment that III-E excludes never reaches (1).
The labeler numbers are not stable across students, one being rated by
labelers 01 to 04 and another by 01, 02, 03 and 05, so the four are taken as
whichever files are present.

**How time is read.** Image names carry a wall-clock timestamp,
`HH_MM_SS_fraction.png`, so t_i and the 1.5 s bins come from the names. The
camera drops frames, which is why a five-minute session holds between 2167
and 2410 images, an effective 7.2 to 8.1 frames per second against a 100 ms
nominal interval. Those dropped stretches are exactly the empty bins III-E
says to flag as missing, and a fixed rate would both hide them and place
frames in the wrong bins.

**Which session is first.** The three activities were filmed as 03 at
12:41, then 08 at 12:46, then 05 at 13:13, so the numeric order is not the
order of recording. Equation (6) wants the earliest session, which is
activity 03, and the order is derived from the timestamps rather than
supplied in a file.

## What you still have to supply

**The data itself.** Unpack each `subject_NN.zip` into a directory named
after it, so the tree reads
`group_02/experiment_03/subject_02/images/...`, and pass the root as
`--dipser-root`. Check with `--describe-only` before a full run.

**Confirmation of the four pending definitions.** Section III-B2 leaves the
face center, shoulder tilt, neck tilt and ear asymmetry as `[PENDING]`.
`tare/geometry.py` implements the anatomical descriptions supplied by the
author, and each function records which description it came from and why it
is written that way. Two constraints from the paper narrowed the choices: an
angle has to survive a scaling and a translation for Proposition 1 to hold,
and the face center has to be a function of the eleven keypoints for
Remark 3 to be the algebraic identity it claims to be. Read those four
docstrings and confirm they say what you meant.

## Things to reconcile in the paper

**A6 names the wrong group.** See above. One word.

**Table III gives 8.7 frames per second.** The sessions measure 7.2 to 8.1,
and the figure is descriptive rather than a parameter, since nothing
computes with it. `config.NOMINAL_FRAME_RATE` records it and
`report_sessions` prints the measured range next to it.

**B4 assumes a fixed panel of four experts.** Equation (7) writes y^(r) as
though expert r were one person across the cohort. The labeler numbering
changes between students, so position r means the r-th labeler file of that
session. The quantity B4 measures, one rater against the mean of the other
three, is still well defined per frame, but the wording implies more than
the data holds.

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
