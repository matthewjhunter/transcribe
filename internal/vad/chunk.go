package vad

import "time"

// Chunk is a contiguous audio range to submit to the ASR backend as a
// single request. Boundaries are absolute times in the original audio,
// matching the time base used for diarization output and for the final
// aligned transcript.
type Chunk struct {
	Start time.Duration
	End   time.Duration
}

// PlanOptions controls how Plan turns Silero segments into upload-ready chunks.
type PlanOptions struct {
	// MinSilence is the silence gap below which adjacent segments may
	// be merged into a single chunk. Default 500ms.
	MinSilence time.Duration

	// MaxChunk is the upper bound on chunk length. Plan never produces
	// a merged chunk longer than this, and hard-splits any single
	// source segment that exceeds it. Default 28s.
	MaxChunk time.Duration

	// MinChunk is the floor on kept chunk length. Chunks shorter than
	// this after merging are dropped. Default 1s.
	MinChunk time.Duration
}

// Defaults for PlanOptions, applied per-field when zero.
const (
	defaultMinSilence = 500 * time.Millisecond
	defaultMaxChunk   = 28 * time.Second
	defaultMinChunk   = 1 * time.Second
)

// Plan turns Silero segments into upload-ready chunks per the policy:
//
//  1. Hard-split any source segment longer than MaxChunk into equal pieces.
//  2. Greedy-merge adjacent pieces whose gap is below MinSilence and whose
//     merged span fits inside MaxChunk.
//  3. Drop chunks shorter than MinChunk.
//
// Returns nil for an empty input.
func Plan(segments []Segment, opts PlanOptions) []Chunk {
	if len(segments) == 0 {
		return nil
	}
	if opts.MinSilence <= 0 {
		opts.MinSilence = defaultMinSilence
	}
	if opts.MaxChunk <= 0 {
		opts.MaxChunk = defaultMaxChunk
	}
	if opts.MinChunk <= 0 {
		opts.MinChunk = defaultMinChunk
	}

	splits := hardSplit(segments, opts.MaxChunk)
	merged := mergeAdjacent(splits, opts.MinSilence, opts.MaxChunk)
	return dropShort(merged, opts.MinChunk)
}

func hardSplit(segments []Segment, maxChunk time.Duration) []Chunk {
	out := make([]Chunk, 0, len(segments))
	for _, s := range segments {
		span := s.End - s.Start
		if span <= maxChunk {
			out = append(out, Chunk(s))
			continue
		}
		// Ceiling division: enough equal pieces that none exceeds maxChunk.
		n := int((span + maxChunk - 1) / maxChunk)
		piece := span / time.Duration(n)
		for i := range n {
			start := s.Start + time.Duration(i)*piece
			end := start + piece
			if i == n-1 {
				end = s.End
			}
			out = append(out, Chunk{Start: start, End: end})
		}
	}
	return out
}

func mergeAdjacent(chunks []Chunk, minSilence, maxChunk time.Duration) []Chunk {
	if len(chunks) == 0 {
		return nil
	}
	out := make([]Chunk, 0, len(chunks))
	cur := chunks[0]
	for _, next := range chunks[1:] {
		gap := next.Start - cur.End
		mergedSpan := next.End - cur.Start
		if gap < minSilence && mergedSpan <= maxChunk {
			cur.End = next.End
			continue
		}
		out = append(out, cur)
		cur = next
	}
	out = append(out, cur)
	return out
}

func dropShort(chunks []Chunk, minChunk time.Duration) []Chunk {
	out := make([]Chunk, 0, len(chunks))
	for _, c := range chunks {
		if c.End-c.Start >= minChunk {
			out = append(out, c)
		}
	}
	return out
}
