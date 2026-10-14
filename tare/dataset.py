"""Reading DIPSER and turning it into the scored frames of Section III-E.

Four things are fixed by the paper. One desk camera per student at 640x480
and about 8.7 frames per second, so every frame holds exactly one student
and the classroom cameras are not touched (assumption A3). Recordings are
sampled in bins of 1.5 s with the first frame of a bin kept, and a bin with
no frame is flagged missing and never filled from a neighbour. Each expert
rating is stored as change points, so it is forward-filled and then
back-filled to one rating per second. The target is the mean of the four
experts, (1).

The student self-assessment does not enter (1). Neither do the age, gender,
emotion and ethnicity estimates in the dataset metadata, nor the smartwatch
signals, by III-B2, and the released pose annotations are unused by III-B1.
The five-vote majority label is not read either: III-E keeps it as a
secondary target but marks the secondary analysis as pending, and no measure
in Section III uses it.

The file names in `DipserLayout` are settings and not facts. Run
`describe_layout` against the release first and adjust the patterns until
the counts match what you expect.
"""

import csv
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

    `time_s` is t_i, seconds from the start of the session. `landmarks` is
    filled in by the extraction pass and is None until then.
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


def describe_layout(layout):
    """Report what the layout patterns actually match, per student and activity.

    Call this before loading anything. It touches no image and reads no
    rating, so it is cheap, and it is the only way to tell a wrong glob from
    an empty release.
    """
    report = {}
    for activity in config.ACTIVITIES:
        for student in _discover_students(layout, activity):
            frames = _frame_paths(layout, activity, student)
            ratings = _rating_paths(layout, activity, student)
            report[(student, activity)] = {
                "frames": len(frames),
                "rating_files": len(ratings),
            }
    return report


def load_cohort(layout):
    """The scored frames of the 16 students of assumption A6.

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

    frames = []
    for student in students:
        for activity in config.ACTIVITIES:
            frames.extend(_load_session(layout, student, activity))
    if not frames:
        raise FileNotFoundError("Students were found but no frame survived sampling.")
    return students, frames


def _load_session(layout, student, activity):
    paths = _frame_paths(layout, activity, student)
    if not paths:
        return []
    ratings = _load_ratings(layout, student, activity)
    if not ratings:
        raise FileNotFoundError(
            f"No expert rating read for student {student}, activity {activity}. "
            "Check rating_glob, rating_time_column and rating_value_column."
        )

    kept = _sample_bins(paths)
    frames = []
    for time_s, path in kept:
        target = _target_at(ratings, int(math.floor(time_s)))
        if target is None:
            # The rating series does not cover this second even after the
            # fill of III-E, so there is no label and the frame is not scored.
            continue
        frames.append(Frame(student, activity, time_s, path, target))
    return frames


def _sample_bins(paths):
    """Keep the first frame of every 1.5 s bin, and report empty bins.

    An empty bin is flagged missing rather than filled from a neighbour, so
    it simply contributes no frame.
    """
    indexed = sorted((_frame_time(p), p) for p in paths)
    bin_width = config.SAMPLING_BIN_SECONDS
    last_time = indexed[-1][0]
    n_bins = int(math.floor(last_time / bin_width)) + 1

    first_in_bin = {}
    for time_s, path in indexed:
        b = int(math.floor(time_s / bin_width))
        if b not in first_in_bin:
            first_in_bin[b] = (time_s, path)

    kept, empty = [], []
    for b in range(n_bins):
        if b in first_in_bin:
            kept.append(first_in_bin[b])
        else:
            empty.append(b)
    _sample_bins.last_empty_bins = empty
    return kept


def _frame_time(path):
    """Seconds from the start of the session, from the frame number.

    DIPSER records at about 8.7 frames per second, so the frame number
    divided by the rate gives t_i.
    """
    numbers = re.findall(r"(\d+)", os.path.basename(path))
    if not numbers:
        raise ValueError(f"No frame number in {path}; set frame_index_from_name = False.")
    return int(numbers[-1]) / config.FRAME_RATE


def _load_ratings(layout, student, activity):
    """One rating per second per expert, forward-filled then back-filled.

    Each expert file holds change points only, so a second between two
    change points takes the value of the earlier one, and a second before
    the first change point takes the value of that first one.
    """
    series = []
    for path in sorted(_rating_paths(layout, activity, student))[: layout.n_experts]:
        points = _read_change_points(path, layout)
        if points:
            series.append(points)
    if len(series) != layout.n_experts:
        raise ValueError(
            f"Expected {layout.n_experts} expert files for student {student}, "
            f"activity {activity} but read {len(series)}. Equation (1) is the "
            "mean of four experts, so a partial set is not averaged."
        )

    last_second = max(max(p) for p in series)
    filled = []
    for points in series:
        filled.append(_fill_forward_then_back(points, last_second))
    return filled


def _read_change_points(path, layout):
    """The change points of one evaluator, as {second: rating}.

    DIPSER stores labels as JSON with the changes marked at the seconds
    where they happen. The exact key names are only knowable from a release
    on disk, so this reader looks for the shape rather than assuming names,
    and raises with the structure printed when it cannot find it. Run
    `inspect_labels` to see that structure and then set `expert_key` and
    `self_evaluation_key` on the layout.
    """
    with open(path) as handle:
        payload = json.load(handle)

    found = _find_second_to_value(payload)
    if found is None:
        raise ValueError(
            f"Could not find a second-to-rating mapping in {path}. "
            f"Top-level structure: {_outline(payload)}. "
            "Run tare.dataset.inspect_labels on this file and set "
            "expert_key on the layout."
        )
    return found


def _find_second_to_value(payload, depth=0):
    """Look for a mapping whose keys read as seconds and whose values read
    as ratings on the 1 to 5 scale of III-C1.
    """
    if depth > 4:
        return None
    if isinstance(payload, dict):
        points = {}
        for key, value in payload.items():
            try:
                second = int(float(key))
            except (TypeError, ValueError):
                points = None
                break
            if not isinstance(value, (int, float)):
                points = None
                break
            points[second] = float(value)
        if points:
            return points
        for value in payload.values():
            found = _find_second_to_value(value, depth + 1)
            if found:
                return found
    elif isinstance(payload, list):
        points = {}
        for entry in payload:
            if not isinstance(entry, dict):
                points = None
                break
            second = entry.get("second", entry.get("time", entry.get("frame")))
            value = entry.get("value", entry.get("attention", entry.get("engagement")))
            if second is None or value is None:
                points = None
                break
            points[int(float(second))] = float(value)
        if points:
            return points
        for entry in payload:
            found = _find_second_to_value(entry, depth + 1)
            if found:
                return found
    return None


def _outline(payload, depth=0):
    """A short description of a JSON structure, for an error message."""
    if depth > 2:
        return "..."
    if isinstance(payload, dict):
        items = list(payload.items())[:6]
        inner = ", ".join(f"{k!r}: {_outline(v, depth + 1)}" for k, v in items)
        more = ", ..." if len(payload) > 6 else ""
        return "{" + inner + more + "}"
    if isinstance(payload, list):
        head = _outline(payload[0], depth + 1) if payload else ""
        return f"[{head}, ... x{len(payload)}]"
    return type(payload).__name__


def inspect_labels(layout, student=None, activity=None):
    """Print the structure of one label file and the evaluator files found.

    Section III-E takes the mean of four experts and leaves the student
    self-assessment out, and DIPSER stores five evaluations per student, so
    which file or key is the self-evaluation has to be identified before the
    target is correct. This prints what is there so that choice is made on
    the real files.
    """
    activity = activity or config.LECTURE_ACTIVITY
    if student is None:
        candidates = _discover_students(layout, activity)
        if not candidates:
            print("No student matched. Check root and frame_glob.")
            return
        student = candidates[0]

    paths = sorted(_rating_paths(layout, activity, student))
    print(f"student {student}, activity {activity}: {len(paths)} label files")
    for path in paths:
        print("  ", os.path.relpath(path, layout.root))
    if not paths:
        return

    with open(paths[0]) as handle:
        payload = json.load(handle)
    print("structure of the first file:")
    print("  ", _outline(payload))
    found = _find_second_to_value(payload)
    if found:
        seconds = sorted(found)
        print(f"  change points found: {len(found)}, "
              f"seconds {seconds[0]} to {seconds[-1]}, "
              f"values {sorted(set(found.values()))}")
    else:
        print("  no second-to-rating mapping recognised in this file")


def _fill_forward_then_back(points, last_second):
    seconds = sorted(points)
    filled = {}
    current = None
    for second in range(0, last_second + 1):
        if second in points:
            current = points[second]
        if current is not None:
            filled[second] = current
    first_value = points[seconds[0]]
    for second in range(0, seconds[0]):
        filled[second] = first_value          # back-fill
    return filled


def _target_at(ratings, second):
    """y_i of (1): the mean of the four experts at this second."""
    values = [r[second] for r in ratings if second in r]
    if len(values) != len(ratings):
        return None
    return float(np.mean(values))


def session_order(layout, students):
    """The sessions of each student, earliest first.

    Stage H2 calibrates on the first 60 s of the first session, "the
    earliest session in time whichever activity it records", so this needs
    real recording times. They are not derivable from the frame numbers,
    which restart each session, so `layout.session_order_file` has to supply
    them: a CSV with columns student, activity and start, where start sorts
    chronologically.
    """
    path = layout.session_order_file
    if not path:
        raise ValueError(
            "session_order_file is empty. Equation (6) calibrates on the "
            "earliest session of each student, so the recording order has to "
            "come from the dataset. Supply a CSV with columns "
            "student, activity, start."
        )
    full = path if os.path.isabs(path) else os.path.join(layout.root, path)
    starts = {}
    with open(full, newline="") as handle:
        for row in csv.DictReader(handle):
            starts[(row["student"], row["activity"])] = row["start"]

    order = {}
    for student in students:
        sessions = [(starts[(student, a)], a) for a in config.ACTIVITIES
                    if (student, a) in starts]
        if not sessions:
            raise ValueError(f"No recording time for any session of student {student}.")
        order[student] = [a for _, a in sorted(sessions)]
    return order


def _discover_students(layout, activity):
    pattern = _fill(layout, layout.frame_glob, activity, "*")
    marker = _fill(layout, layout.frame_glob, activity, "\x00")
    prefix = marker.split("\x00")[0]
    found = set()
    for path in glob.glob(os.path.join(layout.root, pattern)):
        relative = os.path.relpath(path, layout.root)
        if not relative.startswith(prefix):
            continue
        rest = relative[len(prefix):]
        found.add(rest.split(os.sep)[0])
    return sorted(found)


def _fill(layout, pattern, activity, student):
    """Substitute the group, activity and student into a glob pattern.

    The activity and group go through their own patterns, because DIPSER
    directories are not zero-padded while III-E writes the activities as
    03, 05 and 08.
    """
    return pattern.format(
        group=layout.group_dir(),
        activity=layout.activity_dir(activity),
        student=student,
    )


def _frame_paths(layout, activity, student):
    pattern = _fill(layout, layout.frame_glob, activity, student)
    return sorted(glob.glob(os.path.join(layout.root, pattern)))


def _rating_paths(layout, activity, student):
    pattern = _fill(layout, layout.rating_glob, activity, student)
    return sorted(glob.glob(os.path.join(layout.root, pattern)))


def expert_rating_table(layout, students):
    """Every expert rating per student, activity and second.

    B4 of (7) compares one expert against the mean of the other three, so it
    needs the four series rather than their mean.
    """
    table = {}
    for student in students:
        for activity in config.ACTIVITIES:
            table[(student, activity)] = _load_ratings(layout, student, activity)
    return table
