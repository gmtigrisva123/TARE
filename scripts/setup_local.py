"""Unpack the sampled archives and fetch the landmarker models.

Run this once after downloading sampled_experiment_03.zip,
sampled_experiment_05.zip and sampled_experiment_08.zip out of Drive. It
builds the directory tree the loader expects, pulls the two MediaPipe task
files, and then reports what `describe_layout` can see, so a wrong path
shows up here rather than part way through an extraction pass.

    python scripts/setup_local.py --zips ~/Downloads --dest ~/dipser

The tree it writes is

    <dest>/group_02/experiment_03/subject_02/images/*.png
                                            /labels/labeler_*.json

which is what --dipser-root points at.
"""

import argparse
import os
import sys
import shutil
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tare import config, dataset   # noqa: E402

MODELS = {
    "pose_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", required=True,
                        help="Directory holding the three sampled_*.zip files, "
                             "or the wrapper archives Drive hands back when a "
                             "folder is downloaded.")
    parser.add_argument("--dest", required=True,
                        help="Where to build the tree. Becomes --dipser-root.")
    parser.add_argument("--models", default="models",
                        help="Where to put the two MediaPipe task files.")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Leave the archives in place after unpacking.")
    return parser.parse_args()


def find_archive(zips_dir, activity, scratch):
    """Locate one experiment archive, unwrapping a Drive bundle if needed.

    Downloading a folder from Drive gives back one or more wrapper archives
    rather than the files themselves, so the archive wanted here may be
    sitting inside one of them. Any wrapper holding it is opened and the
    inner archive pulled out once.
    """
    name = f"sampled_experiment_{activity}.zip"

    direct = os.path.join(zips_dir, name)
    if os.path.exists(direct):
        return direct, False

    pulled = os.path.join(scratch, name)
    if os.path.exists(pulled):
        return pulled, True

    for wrapper in sorted(os.listdir(zips_dir)):
        if not wrapper.endswith(".zip"):
            continue
        path = os.path.join(zips_dir, wrapper)
        with zipfile.ZipFile(path) as archive:
            inner = [n for n in archive.namelist() if os.path.basename(n) == name]
            if not inner:
                continue
            print(f"    found {name} inside {wrapper}")
            os.makedirs(scratch, exist_ok=True)
            with archive.open(inner[0]) as source, open(pulled, "wb") as sink:
                shutil.copyfileobj(source, sink)
            return pulled, True

    raise FileNotFoundError(
        f"{name} is neither in {zips_dir} nor inside any archive there. "
        "Download the sampled archives from Drive first."
    )


def unpack(zips_dir, dest, keep):
    """Unpack each experiment archive under its own experiment directory."""
    group = f"group_{config.DIPSER_GROUP}"
    scratch = os.path.join(dest, "_unwrapped")
    total = 0

    for activity in config.ACTIVITIES:
        source, was_wrapped = find_archive(zips_dir, activity, scratch)
        target = os.path.join(dest, group, f"experiment_{activity}")
        os.makedirs(target, exist_ok=True)
        with zipfile.ZipFile(source) as archive:
            members = archive.namelist()
            archive.extractall(target)
        total += len(members)
        print(f"  experiment_{activity}: {len(members)} files -> {target}")

        if not keep or was_wrapped:
            # An archive pulled out of a Drive wrapper is a copy, so removing
            # it costs nothing: the wrapper still holds the original.
            os.remove(source)
    if os.path.isdir(scratch) and not os.listdir(scratch):
        os.rmdir(scratch)
    return total


def fetch_models(models_dir):
    """Download the pose and face task files if they are not already there.

    A Python installed outside the system one often has no root certificates
    of its own, so the certifi bundle is used when it is available. Without
    it the download fails on certificate verification rather than on
    anything to do with the file.
    """
    os.makedirs(models_dir, exist_ok=True)

    try:
        import certifi
        import ssl
        context = ssl.create_default_context(cafile=certifi.where())
        opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context)
        )
    except ImportError:
        opener = urllib.request.build_opener()

    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(models_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"  {name}: already here ({os.path.getsize(path) / 2**20:.1f} MB)")
            paths[name] = path
            continue

        print(f"  {name}: downloading")
        try:
            with opener.open(url) as response, open(path, "wb") as sink:
                shutil.copyfileobj(response, sink)
        except Exception as error:
            if os.path.exists(path):
                os.remove(path)
            raise SystemExit(
                f"Could not download {name}: {error}\n"
                f"Fetch it by hand into {models_dir}:\n  curl -L -o "
                f"{os.path.join(models_dir, name)} {url}"
            )
        print(f"    {os.path.getsize(path) / 2**20:.1f} MB")
        paths[name] = path
    return paths


def check(dest):
    """Report what the loader can see, per student and activity."""
    layout = config.DipserLayout(root=dest)
    report = dataset.describe_layout(layout)
    if not report:
        print("  nothing matched. Check --dest against the tree that was written.")
        return False

    students = sorted({s for s, _ in report})
    print(f"  {len(students)} students: {', '.join(students)}")

    ok = True
    for activity in config.ACTIVITIES:
        rows = [(s, report[(s, activity)]) for s, a in report if a == activity]
        frames = sum(r["frames"] for _, r in rows)
        bad_labels = [s for s, r in rows if r["rating_files"] != 4]
        print(f"  experiment_{activity}: {len(rows)} students, {frames} frames")
        if bad_labels:
            ok = False
            print(f"    these students do not have four labeler files: {bad_labels}")

    if len(students) != 16:
        ok = False
        print(f"  assumption A6 names 16 students and this tree has "
              f"{len(students)}.")
    return ok


def main():
    args = parse_args()

    print("unpacking")
    unpack(args.zips, args.dest, args.keep_zips)

    print("\nlandmarker models")
    models = fetch_models(args.models)

    print("\nchecking the tree")
    ok = check(args.dest)

    print("\n" + "-" * 66)
    if ok:
        print("Ready. The full run is")
        print(f"\n  python run_audit.py \\\n"
              f"      --dipser-root {args.dest} \\\n"
              f"      --pose-task {models['pose_landmarker.task']} \\\n"
              f"      --face-task {models['face_landmarker.task']} \\\n"
              f"      --regressor gb")
    else:
        print("Something above is off. Fix it before running the audit.")
        sys.exit(1)


if __name__ == "__main__":
    main()
