# whisper_xpu 验证进度 — 2026-07-27 (updated)

## 任务原点
"录音按钮点了不转录"

## 全部完成 ✅ (real-time mic → VAD → transcribe → text, CPU)

### Root cause of 0-char (the actual bug)
`engine.cpp` set `wp.detect_language=true` in all transcribe paths. In this whisper.cpp
build, `detect_language=true` means **detect-and-exit** (`whisper.cpp:6845` returns 0
after language detection, NO transcription) — NOT "auto-detect then transcribe".
Fix: `detect_language=false` + `language="auto"` (the `auto` branch still detects, then
continues to transcribe). Verified via bench_vad: tiny CPU `no-VAD: 904 chars, 11 segs`
(was 0). See [[whisper-detect-language-detect-and-exit]].

### Task 1: stop() fast via whisper abort ✅
- `engine.h/cpp`: `transcribe_stream(cb, abort_flag)` + new `transcribe_chunk(pcm, n, abort_flag)`.
  Wire `whisper_full_params.abort_callback` + `encoder_begin_callback` to the atomic flag
  → in-flight chunk aborts in ms.
- `audio_recorder.stop()`: drop the `while(m_transcribing) sleep(50ms)` busy-wait; just
  join (abort makes the thread exit in ms). Join before capture teardown (no UAF —
  OnToggleRecord calls `m_recorder.reset()` the instant stop() returns).
- Verified: `stop`→`stopped` same-second across 3 cycles, `encoder_begin_callback
  returned false - aborting`, no segfault, no UAF.

### Task 2: bench_vad isolation ✅
- bench_vad_run.bat (takes model as %1; copies fresh DLL+exe to app Release dir).
- rb.bat is BROKEN (hardcodes the 15-byte placeholder `ggml-large-v3-q8.bin`).
- Re-downloaded real `ggml-tiny.bin` (77 MB) from huggingface.
- Found the detect_language bug (above) — not a streaming/model issue.

### Task 3: VAD chunking, real-time, no drops ✅
- `audio_recorder.h/.cpp`: **producer/consumer** two-thread design.
  - `vad_loop` (producer): drains ring frame-by-frame → energy/RMS VAD → pushes closed
    speech chunks onto a mutex+CV queue. NEVER blocks on transcription → ring can't
    overflow while transcribe_chunk runs.
  - `transcribe_loop` (consumer): pops chunks → `transcribe_chunk` → `on_text` per phrase.
  - energy VAD params: FRAME=320 (20ms), ENERGY_THRESH=0.0035 RMS, MIN_SILENCE=18
    (360ms gap closes chunk), MAX_SPEECH=8s, MIN_TRANSCRIBE=0.5s.
- Language pinning: `Engine::Impl::detected_language` cached on first chunk (via
  `whisper_full_lang_id`) and pinned thereafter → skips the redundant detection encoder
  pass on every chunk (halves per-chunk cost on CPU).
- Verified with turbo: `transcribe chunk 1: 12 chars`, `chunk 2: 18 chars`; ring max 704
  (44ms) during transcription (was 109696 = 6.8s dropped with the old blocking design);
  queue backpressure 0→1→2→1→0; stop aborts in-flight chunk, no crash.

### Task 4: end-to-end + cleanup ✅
- `main.cpp`: fixed wxLog double-print. Root cause: `wxLogStderr::DoLogText` writes
  stderr, then (Windows GUI app, `HasStderr()==false`) re-emits via
  `wxMessageOutputDebug` which on MSW writes stderr AGAIN → doubled lines in the `2>`
  log. Fix: `wxLogStderr(_dup(_fileno(stderr)))` — a FILE* that isn't `stderr` so the
  debug re-emit branch (`m_fp==stderr`) is skipped, output still lands in the same log.
- `vad_loop`: emin/emax now updated during accumulation (were onset-only).
- Verified: tiny run log is single-spaced (no doubling), emin≠emax, ring drained,
  lang detect zh, stop fast, 2 cycles clean exit 0.

## Latency reality (inherent, not bugs)
- turbo-large-v3-q8_0 on 20-thread CPU: ~7s/chunk (first ~15s incl. one-time lang detect),
  because whisper pads every chunk to a 30s window internally. RTF 0.13 on full audio.
- tiny on CPU: ~1-2s/chunk (first ~2-3s). RTF 0.03. Use tiny for genuine sub-realtime.
- GPU (Arc A770M): crashes with `im2col_sycl half kernel` (GGML_SYCL_F16=no) — deferred,
  independent. When GPU works, 30s-padding cost drops to ~RTF 0.03 → true real-time w/ turbo.

## Key files
- `whisper-xpu-core/engine.h/.cpp` — transcribe_chunk, transcribe_stream(abort), lang pin, detect_language fix
- `whisper-xpu-app/src/audio_recorder.h/.cpp` — producer/consumer VAD chunker
- `whisper-xpu-app/main.cpp` — wxLog dedup
- `bench_vad_run.bat` — build+run bench_vad (model as %1); `run_app_log.bat` / `run_app_tiny.bat` — app w/ log capture

## Build/run (sandbox bash can't drive cmd; use Python subprocess)
`python -c "import subprocess; subprocess.run(['cmd','/c','build_app.bat'])"` — builds app+DLL (MSVC+icpx).
`python -c "import subprocess; subprocess.run(['cmd','/c','run_app_log.bat'])"` — app (turbo) w/ `2> app_run.log`.
See [[builds-runs-via-user-bang]].
