"""Cut the 48 DIPSER archives down to the frames the audit actually scores.

Paste into one Colab cell and run. It reads each subject archive straight
from Drive, keeps the first frame of every 1.5 s bin as III-E requires,
copies the four expert label files beside them, and writes one output
archive per experiment. The result is about 4 GB instead of 45.6, small
enough to pull onto a laptop and run the pipeline locally.

Nothing is fully unpacked. A zip is random-access, so only the roughly 200
images a session contributes are ever read, and the images that are dropped
are never written anywhere. Peak disk is one output archive.

What is kept and what is not:

  images/         only the frames that start a bin
  labels/         the four labeler_* files; self_labeling is left behind
                  because (1) excludes the student self-assessment
  metadata/       dropped. III-B1 re-extracts landmarks rather than reading
                  the stored ones, and III-B2 excludes the age, gender,
                  emotion and ethnicity estimates it also holds
  watch_sensors/  dropped, excluded by III-B2

The bin rule here is the one in tare.dataset._sample_bins, and applying it
twice changes nothing: every kept frame still starts its own bin and the
session still begins at the same instant, so the loader re-derives this
exact selection, including the same count of empty bins. The local run
stays the authority either way.

Safe to rerun. An experiment whose output archive is already in Drive is
skipped.
"""

import math
import os
import re
import shutil
import zipfile

from google.colab import drive

SOURCE = "/content/drive/MyDrive/DIPSER/group_02"
DEST = "/content/drive/MyDrive/DIPSER_sampled"
STAGING = "/content/staging"

EXPERIMENTS = ["experiment_03", "experiment_05", "experiment_08"]
BIN_SECONDS = 1.5
IMAGE_STEM = re.compile(r"^(\d{2})_(\d{2})_(\d{2})_(\d+)$")
LINE = "=" * 72


def image_clock(name):
    """Seconds since midnight, from a name like 12_41_17_013271.png."""
    stem = os.path.splitext(os.path.basename(name))[0]
    match = IMAGE_STEM.match(stem)
    if not match:
        raise ValueError(f"Image name {stem!r} is not HH_MM_SS_fraction.")
    hours, minutes, seconds, fraction = match.groups()
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            + int(fraction) / 10 ** len(fraction))


def choose_frames(names):
    """The first image of every 1.5 s bin, and how many bins came up empty."""
    timed = sorted((image_clock(n), n) for n in names)
    start = timed[0][0]
    relative = [(clock - start, name) for clock, name in timed]
    n_bins = int(math.floor(relative[-1][0] / BIN_SECONDS)) + 1

    first_in_bin = {}
    for time_s, name in relative:
        index = int(math.floor(time_s / BIN_SECONDS))
        if index not in first_in_bin:
            first_in_bin[index] = name

    kept = [first_in_bin[i] for i in range(n_bins) if i in first_in_bin]
    return kept, n_bins, n_bins - len(kept)


def copy_session(source_zip, subject, out_archive):
    """Copy one subject's kept frames and label files into the output."""
    with zipfile.ZipFile(source_zip) as archive:
        names = archive.namelist()
        images = [n for n in names
                  if n.startswith("images/") and n.lower().endswith(".png")]
        labels = [n for n in names
                  if n.startswith("labels/")
                  and os.path.basename(n).startswith("labeler_")]

        if not images:
            raise ValueError(f"No image in {source_zip}.")
        if len(labels) != 4:
            raise ValueError(
                f"{source_zip} has {len(labels)} labeler files, expected 4: "
                f"{[os.path.basename(n) for n in labels]}"
            )

        kept, n_bins, empty = choose_frames(images)
        for name in kept:
            out_archive.writestr(
                f"{subject}/images/{os.path.basename(name)}",
                archive.read(name),
            )
        for name in sorted(labels):
            out_archive.writestr(
                f"{subject}/labels/{os.path.basename(name)}",
                archive.read(name),
            )
    return len(images), n_bins, empty, len(kept)


def run_experiment(experiment):
    """Build one output archive, or skip it when it is already in Drive."""
    final = os.path.join(DEST, f"sampled_{experiment}.zip")
    if os.path.exists(final):
        size = os.path.getsize(final) / 2**30
        print(f"{experiment}: already in Drive ({size:.2f} GB), skipping")
        return None

    sources = sorted(
        name for name in os.listdir(os.path.join(SOURCE, experiment))
        if name.endswith(".zip")
    )
    print(f"{experiment}: {len(sources)} subject archives")

    staged = os.path.join(STAGING, f"sampled_{experiment}.zip")
    totals = {"images": 0, "bins": 0, "empty": 0, "kept": 0}

    # PNG is already compressed, so storing is faster and no larger.
    with zipfile.ZipFile(staged, "w", zipfile.ZIP_STORED) as out_archive:
        for n, name in enumerate(sources, start=1):
            subject = os.path.splitext(name)[0]
            images, bins, empty, kept = copy_session(
                os.path.join(SOURCE, experiment, name), subject, out_archive
            )
            totals["images"] += images
            totals["bins"] += bins
            totals["empty"] += empty
            totals["kept"] += kept
            print(f"  [{n:>2}/{len(sources)}] {subject}: "
                  f"{images} images -> {kept} kept, "
                  f"{bins} bins, {empty} empty")

    size = os.path.getsize(staged) / 2**30
    print(f"  staged {size:.2f} GB, copying to Drive")
    os.makedirs(DEST, exist_ok=True)
    shutil.move(staged, final)
    print(f"  wrote {final}")
    return totals


def main():
    drive.mount("/content/drive")
    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(DEST, exist_ok=True)

    grand = {"images": 0, "bins": 0, "empty": 0, "kept": 0}
    for experiment in EXPERIMENTS:
        print(LINE)
        totals = run_experiment(experiment)
        if totals:
            for key in grand:
                grand[key] += totals[key]
        print()

    print(LINE)
    if grand["kept"]:
        print(f"images read      {grand['images']}")
        print(f"bins             {grand['bins']}")
        print(f"empty bins       {grand['empty']}  "
              f"({100 * grand['empty'] / grand['bins']:.1f}% of bins, "
              "left missing by III-E)")
        print(f"frames kept      {grand['kept']}")
    print("\noutput archives:")
    for experiment in EXPERIMENTS:
        path = os.path.join(DEST, f"sampled_{experiment}.zip")
        if os.path.exists(path):
            print(f"  {os.path.basename(path)}  "
                  f"{os.path.getsize(path) / 2**30:.2f} GB")
    print("\nUnpack each one into group_02/<experiment>/ on the laptop, so the "
          "tree reads group_02/experiment_03/subject_02/images/...")


main()
