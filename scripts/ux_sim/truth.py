"""How the analysis did against synth.py's ground truth: tempo, key, and the
section labels (voiceless sections called "verse", drops found).
usage: truth.py META_JSON DB_PATH"""
import json, sqlite3, sys

meta = json.load(open(sys.argv[1], encoding="utf-8"))
db = sqlite3.connect(sys.argv[2]); db.row_factory = sqlite3.Row
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
song = {r["source_url"].rsplit("/", 1)[-1]: r["id"]
        for r in db.execute("SELECT id, source_url FROM songs")}
bpm_ok = key_ok = voiceless = voiceless_verse = drops = drops_found = 0
for m in meta:
    sid = song.get(str(m["i"]))
    if sid is None:
        continue
    f = db.execute("SELECT bpm, key, mode FROM features WHERE song_id=? AND stem_type='full'",
                   (sid,)).fetchone()
    if f and f["bpm"] and abs(f["bpm"] / m["bpm"] - 1) < 0.03:
        bpm_ok += 1
    if f and f["key"]:
        got = f["key"] + ("m" if (f["mode"] or "").startswith("min") else "")
        key_ok += got == m["key"]
    secs = db.execute("SELECT start_sec, end_sec, label, vocal_presence FROM sections "
                      "WHERE song_id=? ORDER BY section_index", (sid,)).fetchall()
    for s in secs:
        if (s["vocal_presence"] or 0) < 0.1:
            voiceless += 1
            voiceless_verse += s["label"] == "verse"
    for t in m.get("sections", []):
        if t["label"] != "drop":
            continue
        drops += 1
        for s in secs:
            overlap = min(s["end_sec"], t["end"]) - max(s["start_sec"], t["start"])
            if s["label"] == "drop" and overlap > 0.5 * (t["end"] - t["start"]):
                drops_found += 1
                break
print(json.dumps({"tracks": len(meta), "bpm_within_3pct": bpm_ok, "key_exact": key_ok,
                  "voiceless_sections": voiceless, "voiceless_labelled_verse": voiceless_verse,
                  "drops": drops, "drops_found": drops_found}, indent=1))
