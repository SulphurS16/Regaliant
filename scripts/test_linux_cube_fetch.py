"""Validate the authored NVIDIA compiler regression, without a GPU or game data."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('spirv-as') and shutil.which('spirv-val'), 'SPIR-V Tools required')
class CubeFetch(unittest.TestCase):
    def test_cube_fetch_is_invalid_and_face_array_fetch_is_valid(self):
        fixture = Path(__file__).resolve().parents[1] / 'crates/rdna/tests/fixtures/cube-fetch-invalid.spvasm'
        source = fixture.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            for valid in (False, True):
                asm, spv = Path(tmp) / 'fetch.spvasm', Path(tmp) / 'fetch.spv'
                asm.write_text(source.replace('OpTypeImage %f32 Cube', 'OpTypeImage %f32 2D') if valid else source)
                subprocess.run(['spirv-as', '--target-env', 'vulkan1.3', str(asm), '-o', str(spv)], check=True, capture_output=True)
                result = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(spv)], capture_output=True, text=True)
                if valid:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('cannot be Cube', result.stderr)
                    self.assertIn('OpImageFetch', result.stderr)


if __name__ == '__main__':
    unittest.main()
