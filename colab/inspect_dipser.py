"""Look inside the DIPSER archives and answer what Section III leaves open.

Paste this into one Colab cell, run it, and send back everything it prints.
Four things are unsettled and all four are decidable only from the real
files:

  a. the directory layout inside a subject archive, which the glob patterns
     in tare/config.py currently guess at
  b. which of the five evaluators is the student self-assessment, which (1)
     has to leave out, and what shape the label JSON has
  c. the true frame rate, where Table III says about 8.7 fps and the DIPSER
     paper says about 10, which decides what lands in each 1.5 s bin
  d. whether anything records when each session was filmed, which (6) needs
     to find the earliest session of a student

Nothing is extracted except the label and metadata files, which are small.
The images are only listed, never unpacked, so this costs no disk and takes
about a minute.
"""

import collections
import json
import os
import zipfile

from google.colab import drive

ROOT = "/content/drive/MyDrive/DIPSER/group_02"
SUBJECT = "subject_02.zip"
EXPERIMENTS = ["experiment_03", "experiment_05", "experiment_08"]
WORK = "/content/peek"

LINE = "=" * 70


def outline(payload, depth=0, limit=3):
    """A short description of a JSON structure."""
    pad = "  " * depth
    if isinstance(payload, dict):
        keys = list(payload)
        head = f"dict with {len(keys)} keys: {keys[:8]}"
        if len(keys) > 8:
            head += " ..."
        lines = [head]
        if depth < limit:
            for key in keys[:4]:
                lines.append(f"{pad}  {key!r} -> {outline(payload[key], depth + 1, limit)}")
        return "\n".join(lines)
    if isinstance(payload, list):
        head = f"list of {len(payload)}"
        if payload and depth < limit:
            head += f", first item -> {outline(payload[0], depth + 1, limit)}"
        return head
    return f"{type(payload).__name__} = {payload!r}"[:120]


def listing(zip_path):
    """Group the archive's entries by their top folder, without extracting."""
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()

    folders = collections.defaultdict(list)
    for name in names:
        parts = name.split("/")
        top = parts[1] if len(parts) > 2 and parts[0] else parts[0]
        folders[top].append(name)
    return names, folders


def small_files(zip_path, folder, out_dir, cap=6):
    """Pull out a few files from one folder so they can be read."""
    pulled = []
    with zipfile.ZipFile(zip_path) as archive:
        wanted = [
            info for info in archive.infolist()
            if f"/{folder}/" in info.filename and not info.is_dir()
        ]
        for info in sorted(wanted, key=lambda i: i.filename)[:cap]:
            target = os.path.join(out_dir, os.path.basename(info.filename))
            with archive.open(info) as source, open(target, "wb") as sink:
                sink.write(source.read())
            pulled.append((info.filename, info.file_size, target))
    return pulled


def main():
    drive.mount("/content/drive")
    os.makedirs(WORK, exist_ok=True)

    # ---- (a) the layout inside one archive --------------------------------
    print(LINE)
    print("A. LAYOUT INSIDE", EXPERIMENTS[0], SUBJECT)
    print(LINE)

    first = os.path.join(ROOT, EXPERIMENTS[0], SUBJECT)
    names, folders = listing(first)
    print(f"{len(names)} entries in the archive\n")
    print("first 5 entries verbatim, so the path shape is visible:")
    for name in names[:5]:
        print("   ", name)

    print("\nby top-level folder:")
    for folder in sorted(folders):
        entries = folders[folder]
        extensions = collections.Counter(
            os.path.splitext(e)[1].lower() or "(none)" for e in entries
        )
        print(f"  {folder:<16} {len(entries):>6} entries   {dict(extensions)}")
        for sample in entries[:3]:
            print(f"      {os.path.basename(sample)}")

    # ---- (b) the labels ---------------------------------------------------
    print()
    print(LINE)
    print("B. LABELS: structure, and which evaluator is the self-assessment")
    print(LINE)

    label_folder = next(
        (f for f in folders if "label" in f.lower()), None
    )
    if label_folder is None:
        print("No folder with 'label' in its name. Folders present:", sorted(folders))
    else:
        print(f"label folder is {label_folder!r}, "
              f"{len(folders[label_folder])} entries")
        print("every file in it:")
        for name in sorted(folders[label_folder]):
            print("   ", os.path.basename(name))

        pulled = small_files(first, label_folder, WORK, cap=6)
        for name, size, path in pulled:
            print(f"\n--- {os.path.basename(name)}  ({size} bytes) ---")
            try:
                with open(path) as handle:
                    payload = json.load(handle)
            except Exception as error:
                print("   not readable as JSON:", error)
                with open(path) as handle:
                    print("   first 300 chars:", handle.read(300))
                continue
            print(outline(payload))
            print("   raw head:", json.dumps(payload)[:400])

    # ---- (c) the frame rate ----------------------------------------------
    print()
    print(LINE)
    print("C. FRAME RATE: image count against session length")
    print(LINE)

    image_folder = next((f for f in folders if "image" in f.lower()), None)
    if image_folder is None:
        print("No folder with 'image' in its name.")
    else:
        images = sorted(folders[image_folder])
        print(f"{len(images)} images in {image_folder!r}")
        print("first 5 and last 3 names, to show the numbering:")
        for name in images[:5]:
            print("   ", os.path.basename(name))
        print("    ...")
        for name in images[-3:]:
            print("   ", os.path.basename(name))
        print("\nTable III assumes 8.7 fps, the DIPSER paper says about 10.")
        for rate in (8.7, 10.0):
            print(f"  at {rate} fps this session would run "
                  f"{len(images) / rate / 60:.2f} minutes")
        print("Compare those against the last second in the label files above: "
              "whichever rate reproduces it is the real one.")

    # ---- (d) recording times ---------------------------------------------
    print()
    print(LINE)
    print("D. METADATA: is there anything that says when each session was filmed")
    print(LINE)

    meta_folder = next((f for f in folders if "metadata" in f.lower()), None)
    if meta_folder is None:
        print("No folder with 'metadata' in its name.")
    else:
        print(f"metadata folder is {meta_folder!r}, "
              f"{len(folders[meta_folder])} entries")
        for name in sorted(folders[meta_folder])[:12]:
            print("   ", os.path.basename(name))
        pulled = small_files(first, meta_folder, WORK, cap=3)
        for name, size, path in pulled:
            print(f"\n--- {os.path.basename(name)}  ({size} bytes) ---")
            try:
                with open(path) as handle:
                    payload = json.load(handle)
                print(outline(payload, limit=2))
            except Exception:
                with open(path, errors="replace") as handle:
                    print("   first 300 chars:", handle.read(300))

    print("\nzip entry timestamps, which sometimes carry the recording date:")
    for experiment in EXPERIMENTS:
        path = os.path.join(ROOT, experiment, SUBJECT)
        if not os.path.exists(path):
            print(f"  {experiment}: not found")
            continue
        with zipfile.ZipFile(path) as archive:
            infos = [i for i in archive.infolist() if not i.is_dir()]
            stamps = sorted(i.date_time for i in infos)
            counts = len(infos)
        print(f"  {experiment}: {counts} entries, "
              f"earliest {stamps[0]}, latest {stamps[-1]}")

    # ---- cross-check the other two sessions -------------------------------
    print()
    print(LINE)
    print("E. THE SAME SUBJECT IN THE OTHER TWO SESSIONS")
    print(LINE)
    for experiment in EXPERIMENTS:
        path = os.path.join(ROOT, experiment, SUBJECT)
        if not os.path.exists(path):
            print(f"{experiment}: missing")
            continue
        _, folders_here = listing(path)
        summary = {f: len(v) for f, v in sorted(folders_here.items())}
        print(f"{experiment}: {summary}")

    print("\nDone. Send back everything above.")


main()
