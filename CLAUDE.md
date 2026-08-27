# metalscope — notes for Claude

- Build with `swift build` / `swift run metalscope` only. Do **not** use `xcodebuild` —
  broken machine-wide (Exec format error) as of 2026-08-22.
- Keep scope tight: CLI + capture library + JSON trace format. Not a GUI app.
- Peaks must be measured (`calibrate`), never asserted from spec sheets — the
  `ChipPeaks.known` table is a labeled fallback only.
- No cloud AI dependencies; everything runs locally.
- Design doc: docs/ARCHITECTURE.md.
