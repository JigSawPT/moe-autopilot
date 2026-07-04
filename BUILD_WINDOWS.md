# Building on Windows (CUDA 13.2 / MSVC workaround)

If you build the fork on Windows with **nvcc 13.2** and **MSVC 14.44**, the CUDA
`cudafe` stubs break with `error C4003` in the `__cudaLaunch` macro expansion. This is a
toolchain-compatibility issue, unrelated to the MoE-split feature itself.

## The clean path: pin the toolset to CUDA 13.0

The measured numbers were taken with the toolset pinned to `cuda=13.0`, where upstream's
unconditional `/Zc:preprocessor` MSVC flag is harmless:

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -T cuda=13.0 \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
cmake --build build --config Release \
  --target llama-bench llama-server llama-aipc-moe-profile -j 16
```

Note the Visual Studio generator ignores `CMAKE_CUDA_COMPILER` — you must select the
toolkit with `-T cuda=<version>`.

## If you must build with CUDA 13.2 (MSVC 14.44)

The root cause: `ggml/src/ggml-cuda/CMakeLists.txt` unconditionally appends
`/Zc:preprocessor` for MSVC (CCCL 3.2+ requires a standards-compliant preprocessor), and
under nvcc 13.2 the stubs then break with C4003. The fix is a one-line gate so the flag
is only applied for CUDA ≥ 13.2 *and* the failing combination is avoided — gate
`/Zc:preprocessor` on `CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.2"` so a CUDA 13.0
build compiles unchanged, and adjust for your local toolchain as needed.

This is a generic toolchain-compat fix, independent of the MoE-split feature. It is kept
out of the upstream feature branch on purpose (upstream CI does not hit the failing
combination, and llama.cpp asks for one concern per PR) — it would be proposed
separately if at all.

## Sanity check

All four targets should build with exit 0 and no `C4003`:

```
llama-bench.exe
llama-server.exe
llama-cli.exe
llama-aipc-moe-profile.exe
```
