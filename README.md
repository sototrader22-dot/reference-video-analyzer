# reference-video-analyzer

Turn any short-form video (TikTok, Reel, Short, or a local file) into a **replicable breakdown**: cut pace, visual style, loudness, a word-level transcript, and a contact sheet with one frame per scene. Built for creative teams who want to understand *why* a reference video works before they make their own.

```bash
python analyze.py https://example.com/some-video
python analyze.py my-clip.mp4 --language en --out analysis/my-clip
```

## What you get

In `analysis/<slug>/`:

| File | Contents |
|---|---|
| `report.md` | Human-readable summary: pace, visuals, audio, speech, hook, plus a "to replicate" checklist |
| `report.json` | Same data, machine-readable |
| `words.json` | Word-level transcript as `{t0, t1, w}`, ready for caption tools (works with [vertical-ad-composer](https://github.com/sototrader22-dot/vertical-ad-composer)) |
| `contact-sheet.jpg` | One frame per scene in a single low-resolution image |

Metrics include cuts per second, average/shortest/longest shot, a four-segment pacing curve (opening, plateau, re-hook, closing), flash shots, brightness, saturation, dominant colors, motion inside shots, black-bar detection, integrated loudness (LUFS), words per minute, pauses, the first-3-seconds hook, and silent tail.

## Install

Requirements: Python 3.9+ and `ffmpeg`/`ffprobe` on your PATH.

```bash
pip install -r requirements.txt
```

## Options

```
python analyze.py <url|file.mp4> [--out DIR] [--threshold 27] [--model small] [--language en] [--no-audio]
```

- `--threshold`: scene-detection sensitivity. Lower finds more cuts.
- `--model`: faster-whisper model size (`tiny`, `base`, `small`, `medium`...). Runs on CPU with int8.
- `--language`: transcription language code. Auto-detected if omitted.
- `--no-audio`: skip loudness and transcription (faster).

## Notes

- Downloading uses [yt-dlp](https://github.com/yt-dlp/yt-dlp). Sites change often; if a download fails, run `pip install -U yt-dlp`, or download the file yourself and pass the local path.
- This tool is for **analysis and reference**. Do not reuse footage, text, faces or logos you do not own.

## Development

```bash
python -m unittest -v
```

CI runs the tests on Python 3.9, 3.11 and 3.12. Heavy dependencies (OpenCV, PySceneDetect, faster-whisper) are imported lazily, so the unit tests need none of them. Contributions are welcome: see the open issues labelled `good first issue`.

## License

MIT
