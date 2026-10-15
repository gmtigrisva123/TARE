"""Download the DIPSER group_02 cohort into Google Drive, from Colab.

Paste this into one Colab cell and run it. It mounts Drive, then fetches the
48 archives the audit needs: the 16 students of group_02 who have all three
of experiments 03, 05 and 08, which are the lecture, the lecture test and
the robotics experimentation of III-E.

Nothing else is fetched. The other six experiments, the other two groups and
the 135 context_cam archives stay where they are, because assumption A3
leaves the classroom cameras out and III-E uses three activities.

Safe to rerun. A file already in Drive at its full size is skipped, and a
partial file is resumed with a Range request rather than started again.
"""

import os
import time

import requests
from google.colab import drive

# Where the archives land. Everything below DEST mirrors the release layout,
# so DEST itself is what --dipser-root points at later.
DEST = "/content/drive/MyDrive/DIPSER/group_02"

CHUNK = 8 * 1024 * 1024      # 8 MB per write
RETRIES = 4                  # per file, with a growing pause between tries
TIMEOUT = 60

# 16 students of group_02 with all three activities, 48 archives.
FILES = [
    ("experiment_03", "subject_02.zip", "https://download.scidb.cn/download?fileId=1996e6fbc3bc1d4c2d30ae9b07840e31&path=/V5/DIPSER/group_02/experiment_03/subject_02.zip&fileName=subject_02.zip"),
    ("experiment_03", "subject_04.zip", "https://download.scidb.cn/download?fileId=c95d543cc8ab6a55fd6cf0497cfbadc6&path=/V5/DIPSER/group_02/experiment_03/subject_04.zip&fileName=subject_04.zip"),
    ("experiment_03", "subject_05.zip", "https://download.scidb.cn/download?fileId=785287129cc0a9d38af72fc52f54623b&path=/V5/DIPSER/group_02/experiment_03/subject_05.zip&fileName=subject_05.zip"),
    ("experiment_03", "subject_06.zip", "https://download.scidb.cn/download?fileId=8ccd4c162c82d155b202ac7b69d578f9&path=/V5/DIPSER/group_02/experiment_03/subject_06.zip&fileName=subject_06.zip"),
    ("experiment_03", "subject_08.zip", "https://download.scidb.cn/download?fileId=70797d9348b9dc7c1b7269e241ee334c&path=/V5/DIPSER/group_02/experiment_03/subject_08.zip&fileName=subject_08.zip"),
    ("experiment_03", "subject_09.zip", "https://download.scidb.cn/download?fileId=b34d5ab0bb6a3452c2f4b1be4eeef338&path=/V5/DIPSER/group_02/experiment_03/subject_09.zip&fileName=subject_09.zip"),
    ("experiment_03", "subject_10.zip", "https://download.scidb.cn/download?fileId=578b678f386204f51117d60f62516413&path=/V5/DIPSER/group_02/experiment_03/subject_10.zip&fileName=subject_10.zip"),
    ("experiment_03", "subject_11.zip", "https://download.scidb.cn/download?fileId=87a9686f272aa072251ebd0871554722&path=/V5/DIPSER/group_02/experiment_03/subject_11.zip&fileName=subject_11.zip"),
    ("experiment_03", "subject_12.zip", "https://download.scidb.cn/download?fileId=4a0d4b95121f3f13db20c2cab49d9e75&path=/V5/DIPSER/group_02/experiment_03/subject_12.zip&fileName=subject_12.zip"),
    ("experiment_03", "subject_15.zip", "https://download.scidb.cn/download?fileId=773989bd88e8ec2bf336b31a1e9ac59a&path=/V5/DIPSER/group_02/experiment_03/subject_15.zip&fileName=subject_15.zip"),
    ("experiment_03", "subject_16.zip", "https://download.scidb.cn/download?fileId=6b1db57a0b32f164302b7808d3f7799a&path=/V5/DIPSER/group_02/experiment_03/subject_16.zip&fileName=subject_16.zip"),
    ("experiment_03", "subject_17.zip", "https://download.scidb.cn/download?fileId=2f998d64efcd26ec06fbbc7a3242d28b&path=/V5/DIPSER/group_02/experiment_03/subject_17.zip&fileName=subject_17.zip"),
    ("experiment_03", "subject_18.zip", "https://download.scidb.cn/download?fileId=cbea29f4596746223f5ea3d17966bb67&path=/V5/DIPSER/group_02/experiment_03/subject_18.zip&fileName=subject_18.zip"),
    ("experiment_03", "subject_19.zip", "https://download.scidb.cn/download?fileId=04d0c2539c0be21321822f73cbb82591&path=/V5/DIPSER/group_02/experiment_03/subject_19.zip&fileName=subject_19.zip"),
    ("experiment_03", "subject_20.zip", "https://download.scidb.cn/download?fileId=57f73be55d62f0dc8ad4497399e9bd14&path=/V5/DIPSER/group_02/experiment_03/subject_20.zip&fileName=subject_20.zip"),
    ("experiment_03", "subject_21.zip", "https://download.scidb.cn/download?fileId=0ae8d3ffc278f187179e059a5dd13a93&path=/V5/DIPSER/group_02/experiment_03/subject_21.zip&fileName=subject_21.zip"),
    ("experiment_05", "subject_02.zip", "https://download.scidb.cn/download?fileId=febbfc78bb54b55c4c513fb24aaed600&path=/V5/DIPSER/group_02/experiment_05/subject_02.zip&fileName=subject_02.zip"),
    ("experiment_05", "subject_04.zip", "https://download.scidb.cn/download?fileId=ab069bf1748e93283780a784a1c27e24&path=/V5/DIPSER/group_02/experiment_05/subject_04.zip&fileName=subject_04.zip"),
    ("experiment_05", "subject_05.zip", "https://download.scidb.cn/download?fileId=b7324d07678c43d87365a9c24d036cc0&path=/V5/DIPSER/group_02/experiment_05/subject_05.zip&fileName=subject_05.zip"),
    ("experiment_05", "subject_06.zip", "https://download.scidb.cn/download?fileId=1d968918870140a37b081e69f0086b27&path=/V5/DIPSER/group_02/experiment_05/subject_06.zip&fileName=subject_06.zip"),
    ("experiment_05", "subject_08.zip", "https://download.scidb.cn/download?fileId=a5037dcbcd8eaad6d97e233cbd55212c&path=/V5/DIPSER/group_02/experiment_05/subject_08.zip&fileName=subject_08.zip"),
    ("experiment_05", "subject_09.zip", "https://download.scidb.cn/download?fileId=6f16e7a2b9e474d3d83ce264e34a6134&path=/V5/DIPSER/group_02/experiment_05/subject_09.zip&fileName=subject_09.zip"),
    ("experiment_05", "subject_10.zip", "https://download.scidb.cn/download?fileId=2c1e98bd9c939a9e31341d46846ebe81&path=/V5/DIPSER/group_02/experiment_05/subject_10.zip&fileName=subject_10.zip"),
    ("experiment_05", "subject_11.zip", "https://download.scidb.cn/download?fileId=7139d17eaccc05bc6f1ef6fead78523d&path=/V5/DIPSER/group_02/experiment_05/subject_11.zip&fileName=subject_11.zip"),
    ("experiment_05", "subject_12.zip", "https://download.scidb.cn/download?fileId=0fa2f73d71c2be2528accc466e069e7c&path=/V5/DIPSER/group_02/experiment_05/subject_12.zip&fileName=subject_12.zip"),
    ("experiment_05", "subject_15.zip", "https://download.scidb.cn/download?fileId=1bd0d5515484282c5c480a4a6736044c&path=/V5/DIPSER/group_02/experiment_05/subject_15.zip&fileName=subject_15.zip"),
    ("experiment_05", "subject_16.zip", "https://download.scidb.cn/download?fileId=d48a2d39866dc593c524cfb8043ecf26&path=/V5/DIPSER/group_02/experiment_05/subject_16.zip&fileName=subject_16.zip"),
    ("experiment_05", "subject_17.zip", "https://download.scidb.cn/download?fileId=b3d5103eb8bb6e8def95e82a4e79238b&path=/V5/DIPSER/group_02/experiment_05/subject_17.zip&fileName=subject_17.zip"),
    ("experiment_05", "subject_18.zip", "https://download.scidb.cn/download?fileId=5c93b7e2b13c0b7e48e9d9276ac1cc42&path=/V5/DIPSER/group_02/experiment_05/subject_18.zip&fileName=subject_18.zip"),
    ("experiment_05", "subject_19.zip", "https://download.scidb.cn/download?fileId=ebcebc77a37df7e3701a0ace8899c274&path=/V5/DIPSER/group_02/experiment_05/subject_19.zip&fileName=subject_19.zip"),
    ("experiment_05", "subject_20.zip", "https://download.scidb.cn/download?fileId=41b25e7c35c7b83c366c4d8442047400&path=/V5/DIPSER/group_02/experiment_05/subject_20.zip&fileName=subject_20.zip"),
    ("experiment_05", "subject_21.zip", "https://download.scidb.cn/download?fileId=2303744e64a751af9d52e1485a48a8eb&path=/V5/DIPSER/group_02/experiment_05/subject_21.zip&fileName=subject_21.zip"),
    ("experiment_08", "subject_02.zip", "https://download.scidb.cn/download?fileId=fb6dd25385b3af4e33eb25bd87894f18&path=/V5/DIPSER/group_02/experiment_08/subject_02.zip&fileName=subject_02.zip"),
    ("experiment_08", "subject_04.zip", "https://download.scidb.cn/download?fileId=73c734e4f3346ca1a2e66bd3daace72c&path=/V5/DIPSER/group_02/experiment_08/subject_04.zip&fileName=subject_04.zip"),
    ("experiment_08", "subject_05.zip", "https://download.scidb.cn/download?fileId=cd9a5ce153b18c431d1d801031b2928d&path=/V5/DIPSER/group_02/experiment_08/subject_05.zip&fileName=subject_05.zip"),
    ("experiment_08", "subject_06.zip", "https://download.scidb.cn/download?fileId=2f95bd2470ab75298bedf43bbb564faf&path=/V5/DIPSER/group_02/experiment_08/subject_06.zip&fileName=subject_06.zip"),
    ("experiment_08", "subject_08.zip", "https://download.scidb.cn/download?fileId=8d1f805b6a47f6f0925ca5b48411270b&path=/V5/DIPSER/group_02/experiment_08/subject_08.zip&fileName=subject_08.zip"),
    ("experiment_08", "subject_09.zip", "https://download.scidb.cn/download?fileId=49b1afe02ceba3fa46c69dc0ad52099b&path=/V5/DIPSER/group_02/experiment_08/subject_09.zip&fileName=subject_09.zip"),
    ("experiment_08", "subject_10.zip", "https://download.scidb.cn/download?fileId=87f6e4a4cc3dea7320fe5816d84398a7&path=/V5/DIPSER/group_02/experiment_08/subject_10.zip&fileName=subject_10.zip"),
    ("experiment_08", "subject_11.zip", "https://download.scidb.cn/download?fileId=92a99dfd881398ac6ba16e400633291a&path=/V5/DIPSER/group_02/experiment_08/subject_11.zip&fileName=subject_11.zip"),
    ("experiment_08", "subject_12.zip", "https://download.scidb.cn/download?fileId=00bb2205a6214ae0a06e07b340c2d53b&path=/V5/DIPSER/group_02/experiment_08/subject_12.zip&fileName=subject_12.zip"),
    ("experiment_08", "subject_15.zip", "https://download.scidb.cn/download?fileId=bdc060d2936dd00acd572fd19728dc33&path=/V5/DIPSER/group_02/experiment_08/subject_15.zip&fileName=subject_15.zip"),
    ("experiment_08", "subject_16.zip", "https://download.scidb.cn/download?fileId=5a1f517b4db50a1fbe169b150e9b048d&path=/V5/DIPSER/group_02/experiment_08/subject_16.zip&fileName=subject_16.zip"),
    ("experiment_08", "subject_17.zip", "https://download.scidb.cn/download?fileId=0a8a263728a9a42533ae32cb76478c88&path=/V5/DIPSER/group_02/experiment_08/subject_17.zip&fileName=subject_17.zip"),
    ("experiment_08", "subject_18.zip", "https://download.scidb.cn/download?fileId=77806f05d78d978b414bc701e6c226f2&path=/V5/DIPSER/group_02/experiment_08/subject_18.zip&fileName=subject_18.zip"),
    ("experiment_08", "subject_19.zip", "https://download.scidb.cn/download?fileId=9f09e51deefaa133a4df3cc0b724991f&path=/V5/DIPSER/group_02/experiment_08/subject_19.zip&fileName=subject_19.zip"),
    ("experiment_08", "subject_20.zip", "https://download.scidb.cn/download?fileId=a7f7ed5d9fc6c2579f1f9cafc88f04f9&path=/V5/DIPSER/group_02/experiment_08/subject_20.zip&fileName=subject_20.zip"),
    ("experiment_08", "subject_21.zip", "https://download.scidb.cn/download?fileId=718ff7d9f29dbf84c7e90d8b75c51d7e&path=/V5/DIPSER/group_02/experiment_08/subject_21.zip&fileName=subject_21.zip"),
]


def already_complete(path, expected):
    """Whether this file is on disk at its full length."""
    return expected and os.path.exists(path) and os.path.getsize(path) == expected


def remote_size(url):
    """The length the server reports, or None when it does not say."""
    try:
        head = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        value = head.headers.get("content-length")
        return int(value) if value else None
    except requests.RequestException:
        return None


def fetch(url, path):
    """Download one archive, resuming a partial file where possible.

    Returns the number of bytes that came over the wire, so a rerun that
    skips everything reports zero.
    """
    expected = remote_size(url)
    if already_complete(path, expected):
        print(f"  have it already ({expected / 2**30:.2f} GB)")
        return 0

    have = os.path.getsize(path) if os.path.exists(path) else 0
    if have and expected and have > expected:
        # Longer than the server's copy, so the local one is wrong.
        os.remove(path)
        have = 0

    for attempt in range(1, RETRIES + 1):
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True,
                              timeout=TIMEOUT) as response:
                response.raise_for_status()
                resuming = response.status_code == 206
                if have and not resuming:
                    have = 0          # the server ignored the range
                mode = "ab" if (have and resuming) else "wb"

                written = have
                started = time.time()
                with open(path, mode) as handle:
                    for block in response.iter_content(CHUNK):
                        handle.write(block)
                        written += len(block)
                        if expected:
                            share = 100 * written / expected
                            rate = (written - have) / 2**20 / max(time.time() - started, 1)
                            print(f"\r  {share:5.1f}%  "
                                  f"{written / 2**30:.2f} GB  {rate:.1f} MB/s",
                                  end="")
                print()
            if expected and os.path.getsize(path) != expected:
                raise IOError(
                    f"got {os.path.getsize(path)} bytes, expected {expected}"
                )
            return written - have
        except (requests.RequestException, IOError) as error:
            print(f"\n  attempt {attempt} failed: {error}")
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if attempt == RETRIES:
                raise
            time.sleep(5 * attempt)
    return 0


def main():
    drive.mount("/content/drive")
    for experiment in sorted({e for e, _, _ in FILES}):
        os.makedirs(os.path.join(DEST, experiment), exist_ok=True)

    total = 0
    failures = []
    for n, (experiment, name, url) in enumerate(FILES, start=1):
        path = os.path.join(DEST, experiment, name)
        print(f"[{n}/{len(FILES)}] {experiment}/{name}")
        try:
            total += fetch(url, path)
        except Exception as error:
            print(f"  giving up on this one: {error}")
            failures.append(f"{experiment}/{name}")

    print(f"\nFetched {total / 2**30:.1f} GB this run into {DEST}")
    if failures:
        print(f"{len(failures)} archives did not finish. Rerun the cell; "
              "the ones that did finish are skipped.")
        for item in failures:
            print("  ", item)
    else:
        print(f"All {len(FILES)} archives are in place.")


main()
