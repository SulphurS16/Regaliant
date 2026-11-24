"""Writable private copies of immutable relay seeds; never chmod the source."""
import os
from pathlib import Path
import shutil
import stat
import tempfile


def copy_tree(source, destination, **kwargs):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f'Snapshot destination already exists: {destination}')
    # copytree applies source modes (including read-only directory modes) after copying.
    # Restore owner permissions only within the newly created, non-symlink snapshot.
    shutil.copytree(source, destination, **kwargs)
    for directory, dirs, files in os.walk(destination, followlinks=False):
        p = Path(directory)
        p.chmod(stat.S_IMODE(p.stat().st_mode) | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in dirs:
            if (p/name).is_symlink(): raise ValueError(f'Snapshot contains a symlink: {p/name}')
        for name in files:
            f = p/name
            if f.is_symlink(): raise ValueError(f'Snapshot contains a symlink: {f}')
            f.chmod(stat.S_IMODE(f.stat().st_mode) | stat.S_IRUSR | stat.S_IWUSR)
    return destination


def check_writable(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Test actual create/rename/remove rights before starting an expensive GPU session.
    with tempfile.NamedTemporaryFile(prefix='.relay-write-', dir=directory, delete=False) as f:
        path = Path(f.name)
        f.write(b'relay write check\n'); f.flush(); os.fsync(f.fileno())
    renamed = path.with_suffix('.renamed')
    try:
        path.replace(renamed)
    finally:
        path.unlink(missing_ok=True); renamed.unlink(missing_ok=True)
