"""Reference video analyzer: turn a short-form video into a replicable breakdown.

Usage:
    python analyze.py <url|file.mp4> [--out analysis/<slug>] [--threshold 27]
                      [--model small] [--language en] [--no-audio]

Steps: yt-dlp (if URL) -> PySceneDetect (cuts) -> OpenCV (pace, motion, color, black bars)
       -> ffmpeg (loudness) -> faster-whisper (word-level transcript)
       -> report.md + report.json + words.json + contact-sheet.jpg

words.json uses the {t0, t1, w} format, so it can be dropped straight into the "words"
field of a vertical-ad-composer plan.

For analysis and reference only. Do not reuse downloaded footage, text, faces or logos you do not own.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def need_tools():
    if not FFMPEG or not FFPROBE:
        sys.exit("ffmpeg and ffprobe must be installed and on your PATH.")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "ref"


def download(url, folder):
    os.makedirs(folder, exist_ok=True)
    template = os.path.join(folder, "ref.%(ext)s")
    r = subprocess.run([sys.executable, "-m", "yt_dlp", "-f", "bv*[height<=1080]+ba/b", "--merge-output-format", "mp4",
                        "--write-info-json", "--no-playlist", "-o", template, url],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("yt-dlp failed:\n" + r.stderr[-600:])
    return os.path.join(folder, "ref.mp4")


def probe(path):
    j = json.loads(subprocess.run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                                  capture_output=True, text=True).stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    return {"duration": float(j["format"]["duration"]), "width": v["width"], "height": v["height"],
            "fps": round(float(num) / float(den), 2), "has_audio": any(s["codec_type"] == "audio" for s in j["streams"])}


def detect_cuts(path, threshold):
    from scenedetect import detect, ContentDetector
    scenes = detect(path, ContentDetector(threshold=threshold))
    return [(round(a.seconds, 2), round(b.seconds, 2)) for a, b in scenes]


COLOR_NAMES = ["red", "orange", "yellow", "green", "green", "blue", "blue", "violet", "violet", "pink", "pink", "red"]


def visual_metrics(path, scenes):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / 6))  # ~6 samples per second
    prev, motion, bright, sat, black, hues, t_motion = None, [], [], [], 0, [], []
    samples = 0
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        samples += 1
        small = cv2.resize(frame, (108, 192))
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        bright.append(float(hsv[..., 2].mean()))
        sat.append(float(hsv[..., 1].mean()))
        # black bars: nearly black rows/columns at the borders
        if (g[:10].mean() < 12 and g[-10:].mean() < 12) or (g[:, :6].mean() < 12 and g[:, -6:].mean() < 12):
            black += 1
        mask = hsv[..., 1] > 50
        if mask.any():
            hues.extend((hsv[..., 0][mask].astype(int) * 2).tolist()[::7])
        if prev is not None:
            motion.append(float(cv2.absdiff(g, prev).mean()))
            t_motion.append(i / fps)
        prev = g
    cap.release()
    # motion inside a shot: ignore the jumps across cuts
    boundaries = [a for a, _ in scenes[1:]]
    inside = [d for d, t in zip(motion, t_motion) if not any(abs(t - b) < 0.2 for b in boundaries)]
    counts = {}
    for h in hues:
        name = COLOR_NAMES[(h // 30) % 12]
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values()) or 1
    return {
        "mean_brightness": round(float(np.mean(bright)), 1),
        "mean_saturation": round(float(np.mean(sat)), 1),
        "motion_within_shots": round(float(np.mean(inside)), 2) if inside else 0,
        "pct_frames_with_black_bars": round(100 * black / max(1, samples), 1),
        "dominant_colors": sorted(((k, round(100 * v / total)) for k, v in counts.items()), key=lambda x: -x[1])[:3],
    }


def loudness(path):
    r = subprocess.run([FFMPEG, "-nostats", "-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    summary = r.stderr[r.stderr.rfind("Summary:"):]  # the earlier "I:" lines are momentary, not the integrated value
    m = re.search(r"I:\s+(-?[\d.]+) LUFS.*?LRA:\s+([\d.]+) LU", summary, re.S)
    return {"lufs": float(m.group(1)), "lra": float(m.group(2))} if m else None


def transcribe(path, folder, model_name, language):
    from faster_whisper import WhisperModel
    import wave
    import numpy as np
    wav = os.path.join(folder, "audio.wav")
    subprocess.run([FFMPEG, "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000", wav], check=True)
    # Pass audio as an array: faster-whisper's decode_audio can break with some PyAV versions.
    with wave.open(wav) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    segments, info = model.transcribe(audio, language=language, word_timestamps=True, vad_filter=False)
    words, phrases = [], []
    for s in segments:
        phrases.append({"t0": round(s.start, 2), "t1": round(s.end, 2), "text": s.text.strip()})
        for w in s.words or []:
            words.append({"t0": round(w.start, 2), "t1": round(w.end, 2), "w": w.word.strip()})
    os.remove(wav)
    return words, phrases, info.language


def contact_sheet(path, scenes, duration, out, max_cols=8):
    """One frame per scene (taken at the middle), tiled into ONE low-resolution image."""
    tmp = os.path.join(os.path.dirname(out), "_frames")
    os.makedirs(tmp, exist_ok=True)
    times = [(a + b) / 2 for a, b in scenes] or [duration / 2]
    times = times[:48]
    for i, t in enumerate(times):
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf",
                        "scale=160:284:force_original_aspect_ratio=increase,crop=160:284", os.path.join(tmp, f"{i:03d}.jpg")])
    cols = min(max_cols, len(times))
    subprocess.run([FFMPEG, "-y", "-v", "error", "-framerate", "1", "-i", os.path.join(tmp, "%03d.jpg"), "-vf",
                    f"tile={cols}x{-(-len(times) // cols)}", "-frames:v", "1", out], check=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)


def pacing_curve(scenes, duration):
    """Cuts per segment: opening 0-8 %, plateau 8-60 %, re-hook 60-70 %, closing 70-100 %."""
    limits = [("opening 0-8%", 0, .08), ("plateau 8-60%", .08, .60), ("re-hook 60-70%", .60, .70), ("closing 70-100%", .70, 1.0)]
    out = []
    for name, a, b in limits:
        n = sum(1 for s, _ in scenes if a * duration <= s < b * duration)
        out.append({"segment": name, "cuts": n, "cuts_per_s": round(n / max(0.01, (b - a) * duration), 2)})
    return out


def main():
    need_tools()
    ap = argparse.ArgumentParser(description="Break a short-form video down into pace, visuals, audio and transcript.")
    ap.add_argument("source", help="video URL or local file")
    ap.add_argument("--out", help="output folder (default: analysis/<slug>)")
    ap.add_argument("--threshold", type=float, default=27.0, help="scene-detection threshold (lower = more cuts)")
    ap.add_argument("--model", default="small", help="faster-whisper model size")
    ap.add_argument("--language", help="transcription language code (auto-detected if omitted)")
    ap.add_argument("--no-audio", action="store_true", help="skip loudness and transcription")
    a = ap.parse_args()

    is_url = a.source.startswith("http")
    name = a.source.split("/")[-1] if is_url else os.path.splitext(os.path.basename(a.source))[0]
    base = a.out or os.path.join("analysis", slug(name or "ref"))
    os.makedirs(base, exist_ok=True)
    video = download(a.source, base) if is_url else a.source
    info = probe(video)
    scenes = detect_cuts(video, a.threshold)
    dur = info["duration"]
    vis = visual_metrics(video, scenes)
    lengths = [b - a_ for a_, b in scenes] or [dur]
    pace = {"n_cuts": max(0, len(scenes) - 1), "cuts_per_s": round(max(0, len(scenes) - 1) / dur, 2),
            "avg_shot_s": round(sum(lengths) / len(lengths), 2), "shortest_shot_s": round(min(lengths), 2),
            "longest_shot_s": round(max(lengths), 2), "flash_shots_0.14s_or_less": sum(1 for x in lengths if x <= 0.14),
            "curve": pacing_curve(scenes, dur)}
    audio, words, phrases, language = None, [], [], None
    if info["has_audio"] and not a.no_audio:
        audio = loudness(video)
        words, phrases, language = transcribe(video, base, a.model, a.language)
    speech = {}
    if words:
        span = words[-1]["t1"] - words[0]["t0"]
        pauses = sum(1 for x, y in zip(words, words[1:]) if y["t0"] - x["t1"] >= 0.2)
        speech = {"language": language, "words": len(words), "wpm": round(len(words) / span * 60),
                  "first_word_s": words[0]["t0"], "pauses_0.2s_or_more": pauses,
                  "silent_tail_s": round(dur - words[-1]["t1"], 2),
                  "hook_3s": " ".join(w["w"] for w in words if w["t0"] < 3)}
    report = {"source": a.source, "video": video.replace("\\", "/"), "info": info, "pace": pace, "visual": vis,
              "audio": audio, "speech": speech, "scenes": scenes, "transcript": phrases}
    with open(os.path.join(base, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)
    with open(os.path.join(base, "words.json"), "w", encoding="utf-8") as fh:
        json.dump(words, fh, ensure_ascii=False)
    contact_sheet(video, scenes, dur, os.path.join(base, "contact-sheet.jpg"))

    L = [f"# Reference breakdown - {os.path.basename(base)}", "",
         f"Source: {a.source}  ", f"{info['width']}x{info['height']} - {info['fps']} fps - **{dur:.1f} s**", "",
         "## Pace", f"- {pace['n_cuts']} cuts -> **{pace['cuts_per_s']} cuts/s**; average shot {pace['avg_shot_s']} s "
         f"(min {pace['shortest_shot_s']} - max {pace['longest_shot_s']}); flash shots (<=0.14 s): {pace['flash_shots_0.14s_or_less']}"]
    L += [f"- {c['segment']}: {c['cuts']} cuts ({c['cuts_per_s']}/s)" for c in pace["curve"]]
    L += ["", "## Visual", f"- Brightness {vis['mean_brightness']}/255 - saturation {vis['mean_saturation']}/255 - colors: "
          + ", ".join(f"{k} {v}%" for k, v in vis["dominant_colors"]),
          f"- Motion within shots: {vis['motion_within_shots']} (mean gray diff; <2 almost still, >8 lots of motion)",
          f"- Frames with black bars: {vis['pct_frames_with_black_bars']} %"]
    if audio:
        L += ["", "## Audio", f"- {audio['lufs']} LUFS - range {audio['lra']} LU"]
    if speech:
        L += ["", "## Speech", f"- {speech['words']} words - **{speech['wpm']} wpm** - first word at {speech['first_word_s']} s - "
              f"pauses >=0.2 s: {speech['pauses_0.2s_or_more']} - silent tail {speech['silent_tail_s']} s",
              f"- **Hook (first 3 s):** \"{speech['hook_3s']}\"", "", "### Transcript"] + \
             [f"- [{p['t0']}-{p['t1']}] {p['text']}" for p in phrases]
    L += ["", "## To replicate (fill in by reading the contact sheet and the transcript)",
          "- Format (talking head / demo / before-after / direct answer / podcast): ___",
          "- Structure: which seconds does each beat occupy (problem, relief, explanation, product, offer)? ___",
          "- Hook type (question / promise / scene / stat) and angle: ___",
          "- What each shot shows, and where the re-hook lands: ___",
          "- Captions: style, keyword color, position: ___",
          "- Audio: music yes/no, sfx on which words: ___",
          "- What NOT to copy: third-party text, faces, logos, claims your product cannot support: ___",
          "", "Contact sheet: `contact-sheet.jpg` (one frame per scene, in order)."]
    with open(os.path.join(base, "report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(os.path.join(base, "report.md"))


if __name__ == "__main__":
    main()
