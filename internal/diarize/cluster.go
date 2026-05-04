package diarize

import (
	"sort"
	"time"
)

// MergeToTarget folds low-duration speakers into their temporal
// neighbors so the distinct speaker count ends up no larger than
// target+tolerance. Returns turns unchanged when:
//
//   - target <= 0 (caller opted out of the soft target)
//   - the distinct speaker count is already within target+tolerance
//
// The merge keeps the (target+tolerance) speakers with the largest
// total speaking duration. Turns belonging to a dropped speaker are
// reassigned to the nearest surviving speaker, measured by absolute
// time-distance to the dropped turn's interval. Speaker IDs are
// renumbered to a contiguous 0..N-1 sequence in first-appearance order
// so the output isn't full of gaps.
//
// Pure post-processing; doesn't require re-running sherpa.
func MergeToTarget(turns []Turn, target, tolerance int) []Turn {
	if target <= 0 || len(turns) == 0 {
		return turns
	}
	if tolerance < 0 {
		tolerance = 0
	}
	cap := target + tolerance

	dur := map[int]time.Duration{}
	for _, t := range turns {
		dur[t.Speaker] += t.End - t.Start
	}
	if len(dur) <= cap {
		return renumberByFirstAppearance(turns)
	}

	// Keep the `cap` speakers with the longest total speaking duration.
	type entry struct {
		Speaker int
		Dur     time.Duration
	}
	entries := make([]entry, 0, len(dur))
	for s, d := range dur {
		entries = append(entries, entry{s, d})
	}
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Dur != entries[j].Dur {
			return entries[i].Dur > entries[j].Dur
		}
		return entries[i].Speaker < entries[j].Speaker
	})

	keep := make(map[int]bool, cap)
	for i := range cap {
		keep[entries[i].Speaker] = true
	}

	out := make([]Turn, len(turns))
	for i, t := range turns {
		if keep[t.Speaker] {
			out[i] = t
			continue
		}
		out[i] = Turn{
			Start:   t.Start,
			End:     t.End,
			Speaker: nearestKeptSpeaker(turns, i, keep),
		}
	}
	return renumberByFirstAppearance(out)
}

// nearestKeptSpeaker scans outward from index i to find the closest
// turn (by interval-center distance) whose speaker is in keep.
//
// Falls back to keep's first member if no kept turn exists in turns —
// that can only happen when every turn was dropped, which means keep
// is empty too; we return the original speaker ID untouched in that
// degenerate case.
func nearestKeptSpeaker(turns []Turn, i int, keep map[int]bool) int {
	if len(keep) == 0 {
		return turns[i].Speaker
	}
	target := (turns[i].Start + turns[i].End) / 2

	bestIdx := -1
	var bestDist time.Duration
	for j, t := range turns {
		if j == i || !keep[t.Speaker] {
			continue
		}
		c := (t.Start + t.End) / 2
		d := c - target
		if d < 0 {
			d = -d
		}
		if bestIdx < 0 || d < bestDist {
			bestIdx = j
			bestDist = d
		}
	}
	if bestIdx < 0 {
		return turns[i].Speaker
	}
	return turns[bestIdx].Speaker
}

// renumberByFirstAppearance maps speaker IDs to a dense 0..N-1 range
// in the order each speaker is first seen in turns.
func renumberByFirstAppearance(turns []Turn) []Turn {
	remap := map[int]int{}
	out := make([]Turn, len(turns))
	for i, t := range turns {
		newID, ok := remap[t.Speaker]
		if !ok {
			newID = len(remap)
			remap[t.Speaker] = newID
		}
		out[i] = Turn{Start: t.Start, End: t.End, Speaker: newID}
	}
	return out
}
