import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
import linux_snapshot


class Snapshot(unittest.TestCase):
    def test_readonly_seed_stays_readonly_but_copy_can_store_and_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'seed'; dst = root/'copy'
            (src/'pipelines/binaries').mkdir(parents=True)
            f = src/'pipelines/binaries/c.pb'; f.write_bytes(b'old'); f.chmod(0o444)
            for p in (src/'pipelines/binaries', src/'pipelines', src): p.chmod(0o555)
            try:
                linux_snapshot.copy_tree(src,dst)
                linux_snapshot.check_writable(dst/'pipelines/binaries')
                (dst/'pipelines/binaries/new.pb').write_bytes(b'new')
                (dst/'pipelines/binaries/c.pb').write_bytes(b'replaced')
                self.assertEqual(f.read_bytes(),b'old')
                self.assertEqual(stat.S_IMODE(f.stat().st_mode),0o444)
                self.assertEqual(stat.S_IMODE(src.stat().st_mode),0o555)
                self.assertTrue((dst/'pipelines/binaries').stat().st_mode & stat.S_IWUSR)
                with self.assertRaises(ValueError): linux_snapshot.copy_tree(src,dst)
            finally:
                for p in (src,src/'pipelines',src/'pipelines/binaries'): p.chmod(0o755)
