"""Reading DIPSER and turning it into the scored frames of Section III-E.

The release stores one desk camera per student at 640x480, so every frame
holds exactly one student and the classroom cameras are not touched
(assumption A3). Recordings are sampled in bins of 1.5 s with the first
frame of a bin kept, and a bin with no frame is flagged missing and never
filled from a neighbour. Each expert rating is stored as change points, so
it is forward-filled and then back-filled to one rating per second. The
target is the mean of the four experts, (1).

Three things about the real files shape this module.

Image names carry a wall-clock timestamp, `HH_MM_SS_microseconds.png`, so
t_i is read from the name rather than counted off a nominal frame rate.
That matters because the camera drops frames: a five-minute session holds
between 2167 and 2410 images, which is 7.2 to 8.1 effective frames per
second against a 100 ms nominal interval. Those dropped stretches are what
produce the empty bins III-E describes, and a fixed rate would both hide
them and put frames in the wrong bins.

A session's labels sit in `labels/` as five files. Four are named
`labeler_*` and one is `self_labeling`, which (1) leaves out. The labeler
numbers are not the same across students: one student may be rated by
labelers 01 to 04 and another by 01, 02, 03 and 05, so the four are taken
as whichever `labeler_*` files are present rather than by fixed names.

Session order is read from the image timestamps too. The three activities
were recorded as 03, then 08, then 05, which is not their numeric order, so
(6) calibrates on activity 03 because it is first in time and not because
it is first by number.

The student self-assessment does not enter (1). Neither do the age, gender,
emotion and ethnicity estimates in `metadata/`, nor the smartwatch signals
in `watch_sensors/`, by III-B2, and the pose landmarks also stored in
`metadata/` are unused by III-B1. The five-vote majority label is not read
either: III-E keeps it as a secondary target but marks the analysis as
pending, and no measure in Section III uses it.
"""

import glob
import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Frame:
    """One scored frame: a sampled image, its identifiers and its target.

    `time_s` is t_i, seconds from the first image of the session.
    `landmarks` is filled in by the extraction pass and is None until then.
    """

    student: str
    activity: str
    time_s: float
    image_path: str
    target: float
    landmarks: object = None

    @property
    def session(self):
        return (self.student, self.activity)

    @property
    def second(self):
        """The second the rating is read at, floor(t_i) by III-C1."""
        return int(math.floor(self.time_s))


# --- clocks ---------------------------------------------------------------

_IMAGE_STEM = re.compile(r"^(\d{2})_(\d{2})_(\d{2})_(\d+)$")
_LABEL_STAMP = re.compile(r"^(\d{2}):(\d{2}):(\d{2}):(\d+)$")


def _seconds_since_midnight(hours, minutes, seconds, fraction):
    """Combine the parts of a timestamp into seconds.

    The fourth field is written with a varying number of digits, so it is
    read as a decimal fraction rather than assumed to be microseconds.
    """
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(fraction) / 10 ** len(fraction))


def image_clock(path):
    """The wall-clock time of one frame, from its file name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    match = _IMAGE_STEM.match(stem)
    if not match:
        raise ValueError(
            f"Image name {stem!r} is not HH_MM_SS_fraction. The sampling of "
            "III-E reads the time from the name, so an unexpected name "
            "cannot be placed in a bin."
        )
    return _seconds_since_midnight(*match.groups())


def label_clock(stamp):
    """The wall-clock time of one change point, from its datetime field."""
    match = _LABEL_STAMP.match(stamp.strip())
    if not match:
        raise ValueError(f"Label datetime {stamp!r} is not HH:MM:SS:fraction.")
    return _seconds_since_midnight(*match.groups())


# --- loading --------------------------------------------------------------

def describe_layout(layout):
    """Report what the layout patterns actually match, per student and activity.

    Call this before loading anything. It opens no image and reads no
    rating, so it is cheap, and it is the only way to tell a wrong glob from
    an empty release.
    """
    report = {}
    for activity in config.ACTIVITIES:
        for student in _discover_students(layout, activity):
            report[(student, activity)] = {
                "frames": len(_frame_paths(layout, activity, student)),
                "rating_files": len(_rating_paths(layout, activity, student)),
            }
    return report


def load_cohort(layout):
    """The scored frames of the cohort, and the order its sessions were filmed.

    A student is kept when all three activities of `config.ACTIVITIES` are
    present, which is how III-E defines the cohort.
    """
    per_activity = {a: set(_discover_students(layout, a)) for a in config.ACTIVITIES}
    students = sorted(set.intersection(*per_activity.values()))
    if not students:
        raise FileNotFoundError(
            "No student has all three activities under the layout given. Run "
            "describe_layout(layout) and check root and frame_glob."
        )

    frames, starts, diagnostics = [], {}, {}
    for student in students:
        for activity in config.ACTIVITIES:
            session, start, info = _load_session(layout, student, activity)
            frames.extend(session)
            starts[(student, activity)] = start
            diagnostics[(student, activity)] = info

    if not frames:
        raise FileNotFoundError("Students were found but no frame survived sampling.")

    order = {
        student: [
            activity for activity in sorted(
                config.ACTIVITIES, key=lambda a: starts[(student, a)]
            )
        ]
        for student in students
    }
    return students, frames, order, diagnostics


def _load_session(layout, student, activity):
    """One session: its sampled frames, its start time, and what was dropped."""
    paths = _frame_paths(layout, activity, student)
    if not paths:
        raise FileNotFoundError(
            f"No image for student {student}, activity {activity}."
        )

    timed = sorted((image_clock(p), p) for p in paths)
    start = timed[0][0]
    relative = [(clock - start, path) for clock, path in timed]

    kept, empty_bins, n_bins = _sample_bins(relative)
    ratings = _load_ratings(layout, student, activity, start)

    frames, unlabelled = [], 0
    for time_s, path in kept:
        target = _target_at(ratings, int(math.floor(time_s)))
        if target is None:
            unlabelled += 1
            continue
        frames.append(Frame(student, activity, time_s, path, target))

    info = {
        "images": len(paths),
        "span_s": relative[-1][0],
        "effective_fps": len(paths) / relative[-1][0] if relative[-1][0] else float("nan"),
        "bins": n_bins,
        "empty_bins": len(empty_bins),
        "sampled": len(kept),
        "dropped_for_no_label": unlabelled,
        "start_clock": start,
        "n_experts": len(ratings),
    }
    return frames, start, info


def _sample_bins(relative):
    """Keep the first frame of every 1.5 s bin, and report the empty ones.

    An empty bin is flagged missing rather than filled from a neighbour, so
    it simply contributes no frame. These appear wherever the camera dropped
    a stretch of frames.
    """
    width = config.SAMPLING_BIN_SECONDS
    n_bins = int(math.floor(relative[-1][0] / width)) + 1

    first_in_bin = {}
    for time_s, path in relative:
        index = int(math.floor(time_s / width))
        if index not in first_in_bin:
            first_in_bin[index] = (time_s, path)

    kept = [first_in_bin[i] for i in range(n_bins) if i in first_in_bin]
    empty = [i for i in range(n_bins) if i not in first_in_bin]
    return kept, empty, n_bins


def _load_ratings(layout, student, activity, session_start):
    """One rating per second for each of the four experts.

    The files hold change points, and a change point may carry only an
    emotion, so a record without an `attention` field leaves the rating
    where it was. Seconds are counted from the first image of the session,
    the same origin t_i uses.
    """
    paths = sorted(_rating_paths(layout, activity, student))
    paths = [p for p in paths if "self_labeling" not in os.path.basename(p)]
    if len(paths) != layout.n_experts:
        raise ValueError(
            f"Expected {layout.n_experts} expert files for student {student}, "
            f"activity {activity} but found {len(paths)}: "
            f"{[os.path.basename(p) for p in paths]}. Equation (1) is the mean "
            "of four experts, so a partial set is not averaged."
        )

    series = [_read_change_points(path, session_start) for path in paths]
    last_second = max(max(points) for points in series if points)
    return [_fill_forward_then_back(points, last_second) for points in series]


def _read_change_points(path, session_start):
    """The attention change points of one expert, as {second: rating}.

    Records carrying only an emotion are skipped, because they do not change
    the attention rating. Where two changes fall in the same second the later
    one stands, which is what a forward fill from that second onward means.
    """
    with open(path) as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of change points in {path}.")

    points = {}
    for record in payload:
        if "attention" not in record:
            continue
        second = int(math.floor(label_clock(record["datetime"]) - session_start))
        points[max(second, 0)] = float(record["attention"])

    if not points:
        raise ValueError(f"No attention change point in {path}.")
    return points


def _fill_forward_then_back(points, last_second):
    """Carry each change point forward, then cover the head of the session."""
    filled, current = {}, None
    for second in range(0, last_second + 1):
        if second in points:
            current = points[second]
        if current is not None:
            filled[second] = current

    first = min(points)
    for second in range(0, first):
        filled[second] = points[first]          # back-fill
    return filled


def _target_at(ratings, second):
    """y_i of (1): the mean of the four experts at this second."""
    values = [series[second] for series in ratings if second in series]
    if len(values) != len(ratings):
        return None
    return float(np.mean(values))


def expert_rating_table(layout, students, starts):
    """Every expert rating per session, for B4 of (7).

    B4 compares one expert against the mean of the other three, so it needs
    the four series rather than their mean. The labeler numbering is not
    consistent across students, so position r here means the r-th labeler
    file of that session and not one person across the cohort.
    """
    table = {}
    for student in students:
        for activity in config.ACTIVITIES:
            table[(student, activity)] = _load_ratings(
                layout, student, activity, starts[(student, activity)]
            )
    return table


# --- paths ----------------------------------------------------------------

def _fill(layout, pattern, activity, student):
    """Substitute the group, activity and student into a glob pattern."""
    return pattern.format(
        group=layout.group_dir(),
        activity=layout.activity_dir(activity),
        student=student,
    )


def _discover_students(layout, activity):
    marker = _fill(layout, layout.frame_glob, activity, "\x00")
    prefix = marker.split("\x00")[0]
    found = set()
    for path in glob.glob(os.path.join(layout.root,
                                       _fill(layout, layout.frame_glob, activity, "*"))):
        relative = os.path.relpath(path, layout.root)
        if not relative.startswith(prefix):
            continue
        found.add(relative[len(prefix):].split(os.sep)[0])
    return sorted(found)


def _frame_paths(layout, activity, student):
    return sorted(glob.glob(os.path.join(
        layout.root, _fill(layout, layout.frame_glob, activity, student)
    )))


def _rating_paths(layout, activity, student):
    return sorted(glob.glob(os.path.join(
        layout.root, _fill(layout, layout.rating_glob, activity, student)
    )))


def survey(root, max_depth=4, max_names=8):
    """Walk the release and report the directory names actually present.

    The glob patterns guess at names. This guesses at nothing: it lists what
    is on disk, level by level, so the patterns can be set from the real tree.
    """
    if not os.path.isdir(root):
        print(f"Not a directory: {root}")
        return

    def walk(path, depth, prefix):
        if depth > max_depth:
            return
        entries = sorted(os.listdir(path))
        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        files = [e for e in entries if not os.path.isdir(os.path.join(path, e))]

        if dirs:
            shown = dirs[:max_names]
            more = f", ... ({len(dirs)} total)" if len(dirs) > max_names else ""
            print(f"{prefix}{len(dirs)} dirs: {', '.join(shown)}{more}")
        if files:
            extensions = {}
            for name in files:
                key = os.path.splitext(name)[1] or "(none)"
                extensions[key] = extensions.get(key, 0) + 1
            print(f"{prefix}{len(files)} files: "
                  + ", ".join(f"{n} {e}" for e, n in sorted(extensions.items())))

        for name in dirs[:max_names]:
            print(f"{prefix}  {name}/")
            walk(os.path.join(path, name), depth + 1, prefix + "    ")

    print(root)
    walk(root, 1, "  ")
