import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { feedbackKey, keyOf } from "../components/pairs/pairModel";
import { toast } from "../toast";

// Every judgement the user has made, in one place.
//
// Stars are the verdict now: 1-5, they filter the library and they train the
// ranking. The server keeps the old three-way verdict alongside them and
// derives whichever is missing, so a row judged with Discover's ✓/~/✗ still
// shows a star here and a star still reaches the learned scorer as a verdict.
//
// Keyed on the SECTION PAIR, not the song pair — "chorus over drop" and "verse
// over breakdown" are different judgements about the same two records, and the
// server's unique index says so too.

export function useRatings() {
  const [byPair, setByPair] = useState({});     // pair key -> 1..5
  const [verdicts, setVerdicts] = useState({}); // pair key -> love|ok|no
  const [rows, setRows] = useState([]);

  const load = useCallback(async () => {
    try {
      const { feedback } = await api.getPairFeedback();
      const r = {}, v = {};
      for (const f of feedback) {
        r[feedbackKey(f)] = f.rating ?? null;
        v[feedbackKey(f)] = f.verdict;
      }
      setByPair(r);
      setVerdicts(v);
      setRows(feedback);
    } catch { /* a missing verdict list is not worth blocking the screen for */ }
  }, []);

  useEffect(() => { load(); }, [load]);

  // A track's own star is the best any pair it appears in has earned. There is
  // no per-song rating store and this deliberately does not add one: the thing
  // being judged is a pairing, and "how good is this record" is a question the
  // app never asks. Reading the max means starring one great mashup lights the
  // track up in the library, which is what the ★ column is for.
  const bySong = useMemo(() => {
    const out = {};
    for (const f of rows) {
      const stars = f.rating;
      if (!stars) continue;
      for (const id of [f.vocal_song_id, f.inst_song_id]) {
        if (out[id] == null || stars > out[id]) out[id] = stars;
      }
    }
    return out;
  }, [rows]);

  const ratingOf = useCallback((c) => (c ? byPair[keyOf(c)] ?? null : null), [byPair]);
  const verdictOf = useCallback((c) => (c ? verdicts[keyOf(c)] ?? null : null), [verdicts]);

  // Optimistic, with a revert. A star that silently failed to save is worse
  // than one that visibly did not take: the next re-score would train on a
  // judgement the user believes they gave.
  //
  // Re-sending the star already stored CLEARS it. The number keys make a stray
  // judgement cheap, so undoing one has to be as cheap, and the affordance is
  // the star already lit. It lives here rather than in a call site so every
  // StarRating that can write — the dock, the player bar, the partners rail,
  // Studio — undoes the same way. Clearing removes the row, so the verdict the
  // star implied goes with it (see models.delete_pair_feedback).
  const rate = useCallback(async (candidate, stars) => {
    if (!candidate) return;
    const k = keyOf(candidate);
    const prevR = byPair[k] ?? null;
    const prevV = verdicts[k] ?? null;
    if (stars === prevR) {
      setByPair((m) => { const o = { ...m }; delete o[k]; return o; });
      setVerdicts((m) => { const o = { ...m }; delete o[k]; return o; });
      setRows((rs) => rs.filter((f) => feedbackKey(f) !== k));
      try {
        await api.clearPairFeedback({
          vocalSongId: candidate.vocal_song_id,
          instSongId: candidate.inst_song_id,
          vocalSection: candidate.vocal_section_idx ?? null,
          instSection: candidate.inst_section_idx ?? null,
        });
      } catch (e) {
        setByPair((m) => ({ ...m, [k]: prevR }));
        setVerdicts((m) => ({ ...m, [k]: prevV }));
        load();
        toast(`Could not clear that rating: ${e.message}`);
      }
      return;
    }
    setByPair((m) => ({ ...m, [k]: stars }));
    try {
      const body = await api.savePairFeedback({
        vocalSongId: candidate.vocal_song_id,
        instSongId: candidate.inst_song_id,
        rating: stars,
        vocalSection: candidate.vocal_section_idx ?? null,
        instSection: candidate.inst_section_idx ?? null,
      });
      setVerdicts((m) => ({ ...m, [k]: body.verdict }));
      setRows((rs) => {
        const rest = rs.filter((f) => feedbackKey(f) !== k);
        return [{
          vocal_song_id: candidate.vocal_song_id,
          inst_song_id: candidate.inst_song_id,
          vocal_section: candidate.vocal_section_idx ?? null,
          inst_section: candidate.inst_section_idx ?? null,
          rating: stars, verdict: body.verdict,
        }, ...rest];
      });
    } catch (e) {
      setByPair((m) => ({ ...m, [k]: prevR }));
      setVerdicts((m) => ({ ...m, [k]: prevV }));
      toast(`Could not save that rating: ${e.message}`);
    }
  }, [byPair, verdicts, load]);

  return { byPair, bySong, ratingOf, verdictOf, rate, refresh: load,
           count: rows.length };
}
