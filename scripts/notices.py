r"""Writes THIRD_PARTY_NOTICES.txt: the license notices of the third-party code compiled into
regaliant.exe and regaliant-launcher.exe (this project's own code: LICENSE).

    python -I scripts\notices.py

The crates are the ones linked into the two programs for the Windows target (`cargo metadata`
with Cargo.lock: normal dependencies, without proc-macros and build-time crates), each with the
license file it ships; where a crate offers a choice, MIT is used. Then the vendored C code
(LibAtrac9, minimp3, the mimalloc allocator libmimalloc-sys builds, the zstd library zstd-sys
builds), Mesa addrlib's notice for
the surface layout tables, the MIT notice of the earlier native port much of the runtime derives
from, and the Rust standard library's notices (the toolchain's COPYRIGHT-library.html), each
distinct notice text once. The Microsoft C runtime linked statically needs no notice.
"""
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'THIRD_PARTY_NOTICES.txt'
TARGET = 'x86_64-pc-windows-msvc'
PROGRAMS = ('dsport', 'launcher')
# Where a crate offers several licenses, the first of these it offers is used.
PREFERENCE = ('MIT', 'ISC', 'Zlib', '0BSD', 'Apache-2.0', 'BSD-3-Clause')
EXT = r'(\.(md|txt))?'
LICENSE_FILES = {
    'MIT': [rf'licen[cs]e[-_.]?mit{EXT}', rf'mit[-_.]licen[cs]e{EXT}'],
    'ISC': [rf'licen[cs]e[-_.]?isc{EXT}'],
    'Zlib': [rf'licen[cs]e[-_.]?zlib{EXT}'],
    '0BSD': [rf'licen[cs]e[-_.]?0bsd{EXT}'],
    'Apache-2.0': [rf'licen[cs]e[-_.]?apache(-2\.0)?{EXT}'],
    'BSD-3-Clause': [rf'licen[cs]e[-_.]?bsd[-_.]?3[-_.]?clause{EXT}'],
}
# Crates whose license expression joins licenses that all apply: the expression decided on and
# the crate's files that carry them, all included (a new expression fails, to be decided again).
BY_HAND = {
    # (Its own code BSD-3-Clause; the zstd sources' legacy decoders BSD-2-Clause.)
    'zstd-sys': ('BSD-3-Clause AND BSD-2-Clause', ['LICENSE', 'LICENSE.BSD-2-Clause']),
}
# Accepted as well when the crate has a single license.
SINGLE_FILES = [rf'licen[cs]e{EXT}', rf'copying{EXT}']
NOTICE_FILE = rf'notice{EXT}'
# License texts of crates that ship none: iced's from its repository; rspirv and spirv are
# Apache-2.0 with no NOTICE file upstream, so the license text itself.
UPSTREAM_TEXTS = {
    'iced-x86': ('MIT', 'third_party/licenses/iced-x86.txt'),
    'rspirv': ('Apache-2.0', 'third_party/licenses/Apache-2.0.txt'),
    'spirv': ('Apache-2.0', 'third_party/licenses/Apache-2.0.txt'),
}
WIDTH = 78

MIT_TEXT = """\
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


def fail(msg):
    sys.exit(f'notices.py: {msg}')


def read(path):
    text = path.read_text(encoding='utf-8').lstrip('﻿').replace('\r\n', '\n')
    return text.strip('\n') + '\n'


def metadata():
    return json.loads(subprocess.run(
        ['cargo', 'metadata', '--format-version', '1', '--locked', '--filter-platform', TARGET,
         '--manifest-path', str(ROOT / 'Cargo.toml')],
        capture_output=True, check=True).stdout)


def linked_crates(meta):
    pkgs = {p['id']: p for p in meta['packages']}
    nodes = {n['id']: n for n in meta['resolve']['nodes']}

    def proc_macro(p):
        return any('proc-macro' in t['kind'] for t in p['targets'])

    todo = [i for i, p in pkgs.items() if p['name'] in PROGRAMS and p['source'] is None]
    if len(todo) != len(PROGRAMS):
        fail('the programs are not in this workspace')
    seen = set()
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]['deps']:
            if any(k['kind'] is None for k in d['dep_kinds']) and not proc_macro(pkgs[d['pkg']]):
                todo.append(d['pkg'])
    crates = [pkgs[i] for i in seen if pkgs[i]['source'] is not None]
    return sorted(crates, key=lambda p: (p['name'], p['version']))


def license_text(p):
    expr = (p.get('license') or '').strip()
    name = f"{p['name']} {p['version']}"
    if not expr:
        fail(f'{name}: no license field')
    if p['name'] in BY_HAND:
        decided, files = BY_HAND[p['name']]
        if expr != decided:
            fail(f"{name}: '{expr}' is not the expression decided on ('{decided}')")
        d = Path(p['manifest_path']).parent
        return expr, '\n'.join(read(d / f) for f in files)
    if ' AND ' in expr or ' WITH ' in expr:
        fail(f"{name}: '{expr}' needs a decision by hand")
    offered = [o.strip(' ()') for o in re.split(r'\s+OR\s+|/', expr)]
    lic = next((x for x in PREFERENCE if x in offered), None)
    if lic is None:
        fail(f"{name}: none of {PREFERENCE} in '{expr}'")
    d = Path(p['manifest_path']).parent
    files = sorted(f.name for f in d.iterdir() if f.is_file())
    patterns = LICENSE_FILES[lic] + (SINGLE_FILES if len(offered) == 1 else [])
    found = [f for f in files if any(re.fullmatch(x, f, re.I) for x in patterns)]
    if found:
        text = read(d / found[0])
    elif UPSTREAM_TEXTS.get(p['name'], (None,))[0] == lic:
        text = read(ROOT / UPSTREAM_TEXTS[p['name']][1])
    else:
        fail(f'{name}: no {lic} license file among {files}')
    if lic == 'Apache-2.0':
        for f in files:
            if re.fullmatch(NOTICE_FILE, f, re.I):
                text += '\n' + read(d / f)
    return lic, text


def crates_section(crates):
    groups = {}
    for p in crates:
        lic, text = license_text(p)
        groups.setdefault((lic, text), []).append(p)
    parts = []
    for (lic, text), members in groups.items():
        names = '\n'.join(f"{p['name']} {p['version']} ({p['license']})" for p in members)
        head = names + f'\nLicense used: {lic}'
        parts.append(f"{'-' * WIDTH}\n{head}\n{'-' * WIDTH}\n\n{text}")
    return '\n'.join(parts)


def mimalloc_notice(meta):
    """The mimalloc C library libmimalloc-sys builds into dsport.exe (its own MIT notice)."""
    p = next((p for p in meta['packages'] if p['name'] == 'libmimalloc-sys'), None)
    if p is None:
        fail('libmimalloc-sys not in the dependency graph')
    d = Path(p['manifest_path']).parent / 'c_src' / 'mimalloc'
    lic = next((f for f in [d / 'v2' / 'LICENSE', d / 'LICENSE'] if f.is_file()), None)
    if lic is None:
        fail(f'no mimalloc LICENSE under {d}')
    return read(lic)


def zstd_notice(meta):
    """The zstd C library zstd-sys builds into both programs (its own BSD notice)."""
    p = next((p for p in meta['packages'] if p['name'] == 'zstd-sys'), None)
    if p is None:
        fail('zstd-sys not in the dependency graph')
    lic = Path(p['manifest_path']).parent / 'zstd' / 'LICENSE'
    if not lic.is_file():
        fail(f'no zstd LICENSE at {lic}')
    return read(lic)


def rust_std_section():
    """The Rust standard library's notices from the toolchain that builds the programs (its
    COPYRIGHT-library.html): the in-tree entries, then each out-of-tree component once, then
    each distinct notice text once with the components that ship it. Only the components of
    the target's standard library (its sysroot's rlibs). Where a component offers MIT among its
    licenses, MIT is used and only its MIT notice is kept."""
    sysroot = Path(subprocess.run(['rustc', '--print', 'sysroot'], capture_output=True, check=True, text=True).stdout.strip())
    src = sysroot / 'share/doc/rust/COPYRIGHT-library.html'
    if not src.is_file():
        fail(f'missing {src}')
    doc = src.read_text(encoding='utf-8')
    libdir = sysroot / 'lib/rustlib' / TARGET / 'lib'
    linked = {re.sub(r'-[0-9a-f]+$', '', f.stem[3:]).replace('-', '_') for f in libdir.glob('lib*.rlib')}
    if not linked:
        fail(f'no rlibs in {libdir}')

    def plain(s):
        return re.sub(r'[^\x20-\x7e\n]', '', html.unescape(re.sub(r'<[^>]+>', '', s)))

    start = doc.find('id="in-tree-files"')
    split = doc.find('id="out-of-tree-dependencies"')
    if start < 0 or split < 0:
        fail(f'unexpected layout of {src}')
    in_tree = [l.strip() for l in plain(doc[doc.find('>', start) + 1:split]).splitlines() if l.strip()]
    in_tree = [l for l in in_tree if l != 'In-tree files' and not l.startswith('<h2')]
    comps, texts = [], {}
    for block in doc[split:].split('<h3>')[1:]:
        name = plain(block[:block.find('</h3>')]).strip()
        if name.rsplit('-', 1)[0].replace('-', '_') not in linked:
            continue

        def field(k):
            m = re.search(rf'<b>{k}:</b>(.*?)</p>', block, re.S)
            return ' '.join(plain(m.group(1)).split()) if m else ''

        lic = field('License') or '?'
        notices = [(plain(f), html.unescape(b).strip('\n') + '\n') for f, b in re.findall(r'<summary><code>(.*?)</code></summary>\s*<pre>(.*?)</pre>', block, re.S)]
        mit_option = re.search(r'\bMIT\b', lic) and re.search(r'\bOR\b|/', lic)
        if mit_option:
            notices = [n for n in notices if 'mit' in n[0].lower() or 'Permission is hereby granted' in n[1]]
        used = ' (MIT used)' if mit_option else ''
        comps.append(f'{name} - {lic}{used}')
        for fname, text in notices:
            key = ' '.join(text.split())
            texts.setdefault(key, [text, []])[1].append(f'{name} ({fname})')
    if not comps:
        fail(f'no components found in {src}')
    parts = [
        'The Rust standard library is dual-licensed under Apache-2.0 and MIT (MIT is',
        'used). Copyright the Rust Project Developers (https://thanks.rust-lang.org).',
        'Its in-tree files:',
        '',
        *in_tree,
        '',
        'The out-of-tree components it was built with:',
        '',
        *comps,
    ]
    for text, members in texts.values():
        parts.append('\n' + '-' * WIDTH + '\n' + '\n'.join(members) + '\n' + '-' * WIDTH + '\n\n' + text)
    return '\n'.join(parts)


INTRO = """\
Third-party notices: regaliant.exe and regaliant-launcher.exe
-------------------------------------------------------------

Regaliant, an unofficial PC port of Demon's Souls, is free software under the GNU General Public
License, version 3 or (at your option) any later version (LICENSE). Besides its own code,
the two programs contain the third-party code below, used under the licenses shown, all of
which allow it in a program under the GPL version 3. Where a component offers a choice of
licenses, the one used is named.

This file is generated by scripts/notices.py from Cargo.lock and the components' own
license files.
"""


def main():
    meta = metadata()
    sections = [
        ('Rust crates', crates_section(linked_crates(meta))),
        ('mimalloc (built by libmimalloc-sys): the memory allocator', mimalloc_notice(meta)),
        ('zstd (built by zstd-sys): the game read from and written to a .zar archive', zstd_notice(meta)),
        ('LibAtrac9 (third_party/libatrac9): ATRAC9 decoding', read(ROOT / 'third_party/libatrac9/LICENSE')),
        ('minimp3 (third_party/minimp3): MP3 decoding (CC0, no notice required)', read(ROOT / 'third_party/minimp3/LICENSE')),
        ('Mesa addrlib: the gfx10 surface layout tables and logic (crates/runtime/src/gpu/swizzle_tables.rs, tiling.rs)',
         'Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.\n\n' + MIT_TEXT),
        ('The earlier native port much of crates/runtime, crates/sce-elf and crates/rdna derives from',
         'Copyright (c) 2026 SulphurS16 and the Bloodborne native port contributors\n\n' + MIT_TEXT),
        ('The Rust standard library', rust_std_section()),
    ]
    out = [INTRO]
    for n, (title, body) in enumerate(sections, 1):
        out.append(f"\n{'-' * WIDTH}\n{n}. {title}\n{'-' * WIDTH}\n\n{body}")
    OUT.write_text(''.join(out), encoding='utf-8', newline='\n')
    print(f'{OUT.name}: {OUT.stat().st_size} bytes')


if __name__ == '__main__':
    main()
