"""Start a command in its own session, so it outlives this terminal.

macOS has no setsid, and plain nohup leaves the child in the caller's
process group, so an editor or agent session that tears down takes the
child with it. This calls os.setsid in the child before exec, which is what
nohup alone does not do.

    python scripts/detach.py <logfile> <command> [args...]

Prints the new process id.
"""

import subprocess
import sys

if len(sys.argv) < 3:
    raise SystemExit(__doc__)

log_path, command = sys.argv[1], sys.argv[2:]
with open(log_path, "ab", buffering=0) as log:
    child = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,      # os.setsid in the child
        close_fds=True,
    )
print(child.pid)
