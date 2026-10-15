"""Parameter values from Table III of the paper.

Everything here is fixed before the audit runs and is the same for RAW,
TARE-S and TARE-FULL. The "Defined in" column of Table III is given after
each value so a reader can check it against the paper.
"""

from dataclasses import dataclass


# --- III-E, sampling and camera -------------------------------------------

SAMPLING_BIN_SECONDS = 1.5          # first frame of each bin is kept
IMAGE_WIDTH = 640                   # W
IMAGE_HEIGHT = 480                  # H

# Table III records the desk camera as running at about 8.7 frames per
# second. Nothing computes with that figure: image file names carry a
# wall-clock timestamp, so t_i and the 1.5 s bins are read from the names.
# The measured rate also varies by session, between 7.2 and 8.1 frames per
# second on the sessions inspected, because the camera drops frames.
NOMINAL_FRAME_RATE = 8.7

ACTIVITIES = ("03", "05", "08")     # lecture, quiz on phones, robotics practice
LECTURE_ACTIVITY = "03"             # the session the proxies of (14) are read from
DIPSER_GROUP = "01"                 # assumption A6

# --- (4), in-frame body scale ---------------------------------------------

EYE_TERM_WEIGHT = 4.5               # lambda
MEDIAN_WINDOW = 5                   # frames of the same session
MEDIAN_WINDOW_ALIGNMENT = "centered"   # "centered" or "trailing"

# --- (6), stage H2 --------------------------------------------------------

CALIBRATION_SECONDS = 60.0          # first 60 s of the first session

# --- III-C5, regressors and folds -----------------------------------------

SEEDS = (0, 1, 2)                   # Omega
MAX_FOLDS = 4                       # K = min(4, N)

RIDGE_ALPHA = 10.0
GB_MAX_ITER = 300
GB_LEARNING_RATE = 0.05
GB_MAX_LEAF_NODES = 15
GB_L2 = 1.0

PREDICTION_RANGE = (1.0, 5.0)       # predictions are clipped to [1, 5]

# --- III-C6, audit measures -----------------------------------------------

LEAK_GB_MAX_ITER = 200
LEAK_GB_LEARNING_RATE = 0.05
LEAK_MAX_FOLDS = 7                  # K_lec = min(7, N_lec)

N_GROUPS = 3                        # tertiles of each proxy
THRESHOLD = 2.5                     # tau, primary
THRESHOLD_SENSITIVITY = (2.0, 3.0)
LABEL_CLASSES = ((1.0, 2.5), (2.5, 3.5), (3.5, 5.0))   # Q = 3
MIN_CELL_STUDENTS = 3               # cells below this are excluded and listed

# --- III-C7 and III-C8, ZOOM ----------------------------------------------

ZOOM_BASE = 1.25
ZOOM_EXPONENTS = (-3, -2, -1, 1, 2, 3)
VERTICAL_SHIFTS = (-40.0, 40.0)     # pixels, at c = 1
N_WARPED_FRAMES = 60                # n_img
ZOOM_RANGE = (0.64, 1.5625)         # F is log-uniform on this range
N_REPLICATES = 20                   # n_rep

# --- III-C9, inference ----------------------------------------------------

N_BOOTSTRAP = 2000                  # resampling students
EQUIVALENCE_MARGIN = 0.05           # delta_eq, rating points
EQUIVALENCE_LEVEL = 0.90            # 90% bootstrap interval
HOLM_LEVEL = 0.05

VARIANTS = ("RAW", "TARE-S", "TARE-FULL")
REGRESSORS = ("ridge", "gb")
PROXIES = ("P_iod", "P_sh")


def perturbation_grid():
    """The grid J of (c, delta) pairs from III-C7.

    Six zoom levels at no shift, plus two pure shifts. The unperturbed pair
    (1, 0) is not in J; it is the reference the drift is measured against.
    """
    pairs = [(ZOOM_BASE ** k, 0.0) for k in ZOOM_EXPONENTS]
    pairs += [(1.0, d) for d in VERTICAL_SHIFTS]
    return pairs


def zoom_levels():
    """The zoom factors of J, used for the common-factor curve of (20)."""
    return [ZOOM_BASE ** k for k in ZOOM_EXPONENTS]


@dataclass
class DipserLayout:
    """Where the DIPSER files sit and how to read them.

    The release is organised as group_NN / experiment_NN / subject_NN, and
    each subject archive unpacks to `images`, `labels`, `metadata` and
    `watch_sensors`. The defaults below were read off the real files, so
    they should need no change as long as each archive is unpacked into a
    directory named after it. Run `tare.dataset.describe_layout` to confirm.

    Experiments 03, 05 and 08 are the lecture, the lecture test and the
    robotics experimentation, which are the three activities III-E names.

    `rating_glob` matches `labeler_*.json` only, which leaves out
    `self_labeling.json`, the student self-assessment that (1) excludes.
    The labeler numbers differ between students, so the four files are
    taken as whichever are present rather than by fixed names.
    """

    root: str
    frame_glob: str = "group_{group}/experiment_{activity}/subject_{student}/images/*.png"
    rating_glob: str = "group_{group}/experiment_{activity}/subject_{student}/labels/labeler_*.json"
    activity_pattern: str = "{:02d}"   # the release writes experiment_03
    group_pattern: str = "{:02d}"      # and group_02
    n_experts: int = 4                 # self_labeling.json is excluded by the glob

    def activity_dir(self, activity):
        return self.activity_pattern.format(int(activity))

    def group_dir(self):
        return self.group_pattern.format(int(DIPSER_GROUP))
