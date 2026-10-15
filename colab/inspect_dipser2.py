"""Second look: the label JSON, and when each session was filmed.

The first pass settled the layout, named the four experts against the self
assessment, and showed that image file names carry a wall-clock timestamp.
Two things are still open, and the first pass missed them because its file
filter looked for "/labels/" while the archive stores "labels/" with no
leading slash.

  1. the shape of a label file, which decides how the change points of
     III-E are read into one rating per second
  2. whether the three sessions of a student can be ordered in time, which
     (6) needs to find the earliest one

This reads a handful of small JSON files and the first and last image name
of each session. It unpacks no image.
"""

import json
import os
import zipfile

from google.colab import drive

ROOT = "/content/drive/MyDrive/DIPSER/group_02"
EXPERIMENTS = ["experiment_03", "experiment_05", "experiment_08"]
SUBJECTS = ["subject_02.zip", "subject_09.zip", "subject_21.zip"]
LINE = "=" * 70


def read_json_in_zip(zip_path, name):
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(name) as handle:
            return json.load(handle)


def entries_under(zip_path, folder):
    """Names directly under one folder of the archive."""
    with zipfile.ZipFile(zip_path) as archive:
        return sorted(
            info.filename for info in archive.infolist()
            if info.filename.startswith(folder + "/") and not info.is_dir()
        )


def clock(name):
    """Seconds since midnight, from an image name like 12_41_19_347321.png."""
    stem = os.path.splitext(os.path.basename(name))[0]
    hours, minutes, seconds, micros = stem.split("_")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(micros) / 1e6


def hhmmss(value):
    return f"{int(value // 3600):02d}:{int(value % 3600 // 60):02d}:{value % 60:06.3f}"


# ---- 1. the label files -------------------------------------------------

def show_labels():
    print(LINE)
    print("1. LABEL FILES, IN FULL")
    print(LINE)

    path = os.path.join(ROOT, EXPERIMENTS[0], SUBJECTS[0])
    for name in entries_under(path, "labels"):
        payload = read_json_in_zip(path, name)
        text = json.dumps(payload, indent=1, ensure_ascii=False)
        print(f"\n--- {os.path.basename(name)} "
              f"({type(payload).__name__}, {len(payload)} top-level items) ---")
        if len(text) <= 2500:
            print(text)
        else:
            print(text[:1800])
            print(f"   ... [{len(text) - 2400} chars cut] ...")
            print(text[-600:])


def compare_labelers():
    """The same labeler across two sessions, to see what varies.

    Whether a file holds only the seconds where the rating changes, as III-E
    describes, shows up as a short list against a five-minute session.
    """
    print()
    print(LINE)
    print("2. ONE LABELER ACROSS SESSIONS AND SUBJECTS")
    print(LINE)
    for subject in SUBJECTS:
        for experiment in EXPERIMENTS:
            path = os.path.join(ROOT, experiment, subject)
            if not os.path.exists(path):
                print(f"{subject} {experiment}: missing")
                continue
            sizes = []
            for name in entries_under(path, "labels"):
                payload = read_json_in_zip(path, name)
                sizes.append(f"{os.path.basename(name).replace('.json','')}={len(payload)}")
            print(f"{subject} {experiment}: " + "  ".join(sizes))


# ---- 2. session times ---------------------------------------------------

def show_times():
    print()
    print(LINE)
    print("3. SESSION START AND END, FROM THE IMAGE NAMES")
    print(LINE)
    print("If the start times rise with the experiment number, then the "
          "lecture is the earliest session and (6) calibrates on it.\n")

    for subject in SUBJECTS:
        print(f"{subject}")
        for experiment in EXPERIMENTS:
            path = os.path.join(ROOT, experiment, subject)
            if not os.path.exists(path):
                print(f"   {experiment}: missing")
                continue
            images = entries_under(path, "images")
            first, last = clock(images[0]), clock(images[-1])
            span = last - first
            print(f"   {experiment}: {len(images):>5} images   "
                  f"{hhmmss(first)} -> {hhmmss(last)}   "
                  f"span {span:7.2f}s   {len(images) / span:5.2f} fps")
        print()


# ---- 3. one metadata file, to confirm what it holds ---------------------

def show_metadata():
    print(LINE)
    print("4. ONE METADATA FILE")
    print(LINE)
    print("III-B2 excludes the age, gender, emotion and ethnicity estimates "
          "in here, and III-B1 re-extracts landmarks rather than reading "
          "them. This is only to confirm what is being declined.\n")

    path = os.path.join(ROOT, EXPERIMENTS[0], SUBJECTS[0])
    names = entries_under(path, "metadata")
    payload = read_json_in_zip(path, names[0])
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    print(f"--- {os.path.basename(names[0])} ---")
    print(text[:2000])
    if len(text) > 2000:
        print(f"   ... [{len(text) - 2000} more chars]")


drive.mount("/content/drive")
show_labels()
compare_labelers()
show_times()
show_metadata()
print("\nDone.")
