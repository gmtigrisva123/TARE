"""Warp frames and re-extract landmarks across several processes.

The two ZOOM protocols of III-C7 and III-C8 rerun the landmark extractor on
warped images, and at the Table III parameters that comes to roughly a
quarter of a million extractions. One process manages about two a second on
a two-core laptop, so the work is handed to a pool instead.

Splitting it changes no result. Extraction is a pure function of one image:
each frame is warped and read on its own, nothing is shared between frames,
and the output is put back in the caller's order. The parameters stay
exactly as Table III sets them.

A MediaPipe landmarker cannot be sent between processes, so each worker
builds its own once and keeps it for the life of the pool.
"""

import multiprocessing as mp
import os

from . import zoom

_EXTRACTOR = None


def _init_worker(pose_task, face_task):
    """Build this worker's landmarker. Runs once per process."""
    global _EXTRACTOR
    from .landmarks import LandmarkExtractor

    # Each worker is single-threaded; the parallelism is across processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _EXTRACTOR = LandmarkExtractor(pose_task, face_task)


def _warp_and_extract(job):
    """One job: read a frame, warp it, and return z_i for the result."""
    index, path, c, delta = job
    import cv2

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read frame {path}.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return index, _EXTRACTOR(zoom.warp_image(image, c, delta))


class BatchExtractor:
    """Run a list of warp-and-extract jobs, in a pool or in this process.

    `workers = 1` keeps everything here, which is what the tests and any
    debugging run want. Anything larger opens a pool and reuses it, because
    starting one costs a model load per worker.
    """

    def __init__(self, pose_task, face_task, workers=1, progress=None):
        self.pose_task = pose_task
        self.face_task = face_task
        self.workers = max(1, int(workers))
        self.progress = progress
        self._pool = None
        self._local = None

    def __enter__(self):
        if self.workers > 1:
            context = mp.get_context("spawn")
            self._pool = context.Pool(
                processes=self.workers,
                initializer=_init_worker,
                initargs=(self.pose_task, self.face_task),
            )
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __call__(self, jobs):
        """Map (index, path, c, delta) jobs to {index: landmarks}."""
        jobs = list(jobs)
        if not jobs:
            return {}

        results = {}
        if self._pool is None:
            if self._local is None:
                from .landmarks import LandmarkExtractor
                self._local = LandmarkExtractor(self.pose_task, self.face_task)
            global _EXTRACTOR
            _EXTRACTOR = self._local
            iterator = (_warp_and_extract(job) for job in jobs)
        else:
            chunk = max(1, len(jobs) // (self.workers * 8))
            iterator = self._pool.imap_unordered(
                _warp_and_extract, jobs, chunksize=chunk
            )

        for done, (index, landmarks) in enumerate(iterator, start=1):
            results[index] = landmarks
            if self.progress and done % 2000 == 0:
                self.progress(done, len(jobs))
        return results
