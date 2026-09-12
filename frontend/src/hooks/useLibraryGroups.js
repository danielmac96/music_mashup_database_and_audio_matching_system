import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api";

// Library groups: a crate, seen from the library side.
//
// A crate is a shortlist you build on Discover, where the question is "which
// crates hold this SoundCloud permalink" — the track need not be in the library
// at all. Once its items are ingested the same crate answers a different and
// more useful question: "which of my songs are on this shelf". That is a GROUP,
// and it is what makes a saved SoundCloud playlist a smaller library rather than
// a bookmark folder.
//
// Fetched ONCE, here, for the same reason the library and the judgements are:
// the rail draws every group's count, the filter bar names them, the table is
// narrowed by one and the row menu adds to them. Four fetches of the same list
// would be four answers that disagree the moment one of them adds a track.
//
// The membership itself is a set of ids, deliberately: filtering by a group is
// then arithmetic over rows already in memory, exactly like every other library
// filter, and never a request. See hooks/useLibraryFilters.js.

const EMPTY = [];

export function useLibraryGroups() {
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const body = await api.getLibraryGroups();
      setGroups(body.groups || []);
      setError(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  // songId -> the groups holding it, and (groupId, songId) -> its position in
  // that group. The position is what "group order" sorts on: a saved playlist
  // keeps its running order, which is the whole reason you saved it as one
  // rather than as a set of filters.
  const index = useMemo(() => {
    const bySong = new Map();
    const position = new Map();
    for (const g of groups) {
      (g.song_ids || []).forEach((songId, i) => {
        if (!bySong.has(songId)) bySong.set(songId, []);
        bySong.get(songId).push(g.id);
        position.set(`${g.id}:${songId}`, i);
      });
    }
    return { bySong, position };
  }, [groups]);

  const groupsOf = useCallback(
    (songId) => index.bySong.get(songId) || EMPTY, [index]);

  // A string id from a <select> and a number id from the API have to compare
  // equal, or the filter silently matches nothing.
  const has = useCallback((groupId, songId) => {
    const ids = index.bySong.get(songId);
    return !!ids && ids.some((id) => String(id) === String(groupId));
  }, [index]);

  const positionOf = useCallback(
    (groupId, songId) => {
      const at = index.position.get(`${Number(groupId)}:${songId}`);
      return at == null ? null : at;
    }, [index]);

  const membership = useMemo(
    () => ({ has, positionOf }), [has, positionOf]);

  const add = useCallback(async (groupId, songIds) => {
    const res = await api.addSongsToCrate(groupId, songIds);
    await refresh();
    return res;
  }, [refresh]);

  const remove = useCallback(async (groupId, songIds) => {
    const res = await api.removeSongsFromCrate(groupId, songIds);
    await refresh();
    return res;
  }, [refresh]);

  const create = useCallback(async (name) => {
    const made = await api.createCrate(name);
    await refresh();
    return made;
  }, [refresh]);

  const byId = useCallback(
    (groupId) => groups.find((g) => String(g.id) === String(groupId)) || null,
    [groups]);

  return { groups, loading, error, refresh, membership, groupsOf, byId,
           add, remove, create };
}
