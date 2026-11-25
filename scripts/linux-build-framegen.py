#!/usr/bin/env python3
"""Build the optional MIT FSR3 Vulkan backend and Regaliant bridge from pinned source."""
import argparse
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request
import linux_guard
import importlib.util

PIN='2c90b7f4b877b97e5779aa4d09d338c1e29cb75c'
SHA256='b200292e02fd6598ecf915105e8d74dd9da6a25b2f433464058567b81d3bf27d'
URL=f'https://codeload.github.com/ychwch/FidelityFX-SDK-Linux/tar.gz/{PIN}'

def adapter_digest(repo):
    h=hashlib.sha256()
    for name in ('framegen_bridge.cpp','framegen_motion.hpp','framegen_motion.comp'):
        h.update((repo/'crates/runtime/src/platform/linux'/name).read_bytes())
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-only',action='store_true',help='rebuild ABI/shader adapter using the existing pinned SDK and installed backend')
    parser.add_argument('--archive',type=Path,help='reuse a downloaded archive (hash checked)')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parent.parent
    spec=importlib.util.spec_from_file_location('prepare',repo/'scripts/linux-prepare-shaders.py')
    prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)
    game=Path(os.environ.get('DS_GAME_DIR',repo/'local-data/nonexistent-game-root'))
    root=repo/'local-data/linux-port/fsr-framegen'
    prepare.protect_output(root,game,os.environ)
    prepare.protect_output(repo/'target/release/upscalers',game,os.environ)
    root.mkdir(parents=True,exist_ok=True)
    source=root/'source'
    if args.bridge_only and not (source/'.regaliant-source-pin').is_file():
        raise ValueError('--bridge-only requires the existing pinned SDK source')
    archive=args.archive or root/'sdk.tar.gz'
    if not args.bridge_only and not archive.exists():
        if args.archive: raise ValueError('--archive must name an existing source archive')
        tmp=archive.with_suffix('.part')
        with urllib.request.urlopen(URL,timeout=60) as src,tmp.open('wb') as dst: shutil.copyfileobj(src,dst)
        tmp.replace(archive)
    if not args.bridge_only:
        with archive.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=SHA256: raise ValueError('FSR source archive hash mismatch')
    source=root/'source'
    if not source.exists():
        final_source=source;source=root/'source-incomplete'
        if source.exists(): raise ValueError(f'remove incomplete source directory before retry: {source}')
        with tarfile.open(archive) as t:
            prefix=t.getmembers()[0].name+'/'
            for m in t:
                rel=Path(m.name.removeprefix(prefix))
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('unsafe archive path')
                if m.isfile() and (str(rel).startswith('sdk/') or str(rel).startswith('docs/license')) and not str(rel).startswith('sdk/bin/'):
                    dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(t.extractfile(m).read())
        # Bound the native permutation compiler too: upstream uses os.cpu_count(), ignoring affinity.
        compiler=source/'sdk/tools/binary_store/ffx_sc_native.py'
        text=compiler.read_text();old='min(os.cpu_count() or 1, max(1, len(to_compile)))'
        if old not in text: raise ValueError('pinned shader compiler no longer matches worker patch')
        compiler.write_text(text.replace(old,'min(int(os.environ.get("DS_COMPILE_WORKERS", "2")), max(1, len(to_compile)))'))
        (source/'.regaliant-source-pin').write_text(PIN+'\n'+SHA256+'\n')
        source.rename(final_source);source=final_source
    if (source/'.regaliant-source-pin').read_text()!=PIN+'\n'+SHA256+'\n':
        raise ValueError('existing source does not match pinned backend; use a fresh source directory')
    build=root/'build';out=repo/'target/release/upscalers';out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CARGO_HOME=str(repo/'.cargo-home'))
    env.setdefault('DS_RELAY_CPUS',','.join(map(str,linux_guard.select_cpus()[:2])))
    env.setdefault('DS_COMPILE_WORKERS','2');env.setdefault('DS_RELAY_THERMAL_MODE','observe')
    jobs=str(max(1,min(2,int(env['DS_COMPILE_WORKERS']))))
    commands=[['cmake','-S',str(source/'sdk'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DFFX_API_BACKEND=VK_X64','-DFFX_FSR3UPSCALER=ON','-DFFX_SPD=ON','-DFFX_FI=ON','-DFFX_OF=ON','-DFFX_AUTO_COMPILE_SHADERS=ON'],
        ['cmake','--build',str(build),'--target','ffx_backend_vk_x64','-j',jobs]]
    with linux_guard.cpu_heavy_lock():
        for i,cmd in enumerate([] if args.bridge_only else commands):
            with (root/f'build-{i}.log').open('w') as log:
                code=linux_guard.run(['timeout','-k','5s','1800s',*cmd],root/f'build-{i}',env=env,stdout=log,stderr=log)
            if code: raise ValueError(f'FSR build failed ({code}); see {root}/build-{i}.log')
        backend=out/'libffx_backend_vk_x64.so' if args.bridge_only else next(build.rglob('libffx_backend_vk_x64.so'))
        if not backend.is_file(): raise ValueError('missing native SDK backend')
        # Replace inodes, never truncate a library that a running process may have mapped.
        if not args.bridge_only:
            shutil.copy2(backend,out/(backend.name+'.tmp'))
            (out/(backend.name+'.tmp')).replace(out/backend.name)
        # The small input adapter is our shader; the SDK shaders remain unmodified.
        cmd=['glslangValidator','-V','--target-env','vulkan1.1','--vn','regaliant_motion_spv',str(repo/'crates/runtime/src/platform/linux/framegen_motion.comp'),'-o',str(root/'framegen_motion.h')]
        with (root/'motion-build.log').open('w') as log:
            code=linux_guard.run(['timeout','120s',*cmd],root/'motion-build',env=env,stdout=log,stderr=log)
        if code: raise ValueError('motion adapter compile failed; see motion-build.log')
        cmd=['c++','-std=c++20','-O2','-fPIC','-shared','-Wl,-z,defs','-I'+str(source/'sdk/include'),'-I'+str(root),str(repo/'crates/runtime/src/platform/linux/framegen_bridge.cpp'),'-L'+str(out),'-lffx_backend_vk_x64','-lvulkan','-Wl,-rpath,$ORIGIN','-o',str(out/'libregaliant_fsrfg.so.tmp')]
        with (root/'bridge-build.log').open('w') as log:
            code=linux_guard.run(['timeout','120s',*cmd],root/'bridge-build',env=env,stdout=log,stderr=log)
        if code: raise ValueError(f'FSR bridge build failed ({code}); see {root}/bridge-build.log')
        # Linkers permit unresolved references in dependencies. Exercise RTLD_NOW too,
        # without creating an instance/device, before publishing a loadable bridge.
        lib=ctypes.CDLL(str(out/'libregaliant_fsrfg.so.tmp'))
        if lib.regaliant_fsrfg_abi()!=2: raise ValueError('native FSR bridge ABI check failed')
        (out/'libregaliant_fsrfg.so.tmp').replace(out/'libregaliant_fsrfg.so')
        (out/'fsr-framegen-adapter.sha256').write_text(adapter_digest(repo)+'\n')
    shutil.copy2(source/'docs/license.md',out/'fsr-framegen-LICENSE.md')
    (out/'fsr-framegen-source.txt').write_text(f'{URL}\nsha256 {SHA256}\nNative build; worker cap patch only. No downloaded binary used.\n')
    print(out/'libregaliant_fsrfg.so')

if __name__=='__main__': main()
