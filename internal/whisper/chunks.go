package whisper

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/matthewjhunter/transcribe/internal/audio"
	"github.com/matthewjhunter/transcribe/internal/vad"
)

// TranscribeChunks submits each chunk of `samples` (sliced by chunk
// time bounds) as an independent transcription request, then merges the
// per-chunk results into a single Result with global timestamps.
//
// concurrency caps simultaneous in-flight requests; values < 1 are
// treated as 1. Errors fail-fast: the first chunk to fail cancels the
// rest and is returned.
//
// Each chunk gets a fresh decoder state on the backend, so a Whisper
// repetition loop on one chunk cannot poison adjacent chunks.
//
// Pass WithChunkStore to checkpoint completed chunks: a chunk already in
// the store is loaded rather than re-requested, and every fresh chunk is
// saved as it finishes, so a killed run resumes instead of starting over.
func (c *Client) TranscribeChunks(
	ctx context.Context,
	samples []float32, sampleRate int,
	chunks []vad.Chunk,
	concurrency int,
	opts ...ChunkOption,
) (*Result, error) {
	if len(chunks) == 0 {
		return &Result{}, nil
	}
	if concurrency < 1 {
		concurrency = 1
	}

	var opt chunkOptions
	for _, o := range opts {
		o(&opt)
	}

	results := make([]*Result, len(chunks))
	g, gctx := errgroup.WithContext(ctx)
	sem := make(chan struct{}, concurrency)

	for i, ch := range chunks {
		g.Go(func() error {
			key := ChunkKey(ch)

			// A checkpointed chunk is served from the store without
			// touching the backend or a concurrency slot.
			if opt.store != nil {
				if r, ok := opt.store.Load(key); ok {
					results[i] = r
					return nil
				}
			}

			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-gctx.Done():
				return gctx.Err()
			}

			startSample := durationToSamples(ch.Start, sampleRate)
			endSample := durationToSamples(ch.End, sampleRate)
			startSample = max(startSample, 0)
			endSample = min(endSample, len(samples))
			if startSample >= endSample {
				return nil
			}

			var buf bytes.Buffer
			if err := audio.EncodePCM16WAV(samples[startSample:endSample], sampleRate, &buf); err != nil {
				return fmt.Errorf("whisper: chunk %d: encode wav: %w", i, err)
			}
			name := fmt.Sprintf("chunk-%05d.wav", i)
			r, err := c.transcribeBody(gctx, &buf, name)
			if err != nil {
				return fmt.Errorf("whisper: chunk %d: %w", i, err)
			}
			results[i] = offsetResult(r, ch.Start)

			if opt.store != nil {
				if err := opt.store.Save(key, results[i]); err != nil {
					return fmt.Errorf("whisper: chunk %d: checkpoint: %w", i, err)
				}
			}
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}

	merged := mergeResults(results)
	merged.Duration = chunks[len(chunks)-1].End
	return merged, nil
}

func durationToSamples(d time.Duration, sampleRate int) int {
	return int(float64(d) / float64(time.Second) * float64(sampleRate))
}

func offsetResult(r *Result, offset time.Duration) *Result {
	out := &Result{
		Text:     r.Text,
		Language: r.Language,
		Segments: make([]Segment, len(r.Segments)),
		Words:    make([]Word, len(r.Words)),
	}
	for i, s := range r.Segments {
		out.Segments[i] = Segment{
			Start: s.Start + offset,
			End:   s.End + offset,
			Text:  s.Text,
		}
	}
	for i, w := range r.Words {
		out.Words[i] = Word{
			Start: w.Start + offset,
			End:   w.End + offset,
			Text:  w.Text,
		}
	}
	return out
}

func mergeResults(parts []*Result) *Result {
	out := &Result{}
	var sb strings.Builder
	for _, p := range parts {
		if p == nil {
			continue
		}
		if t := strings.TrimSpace(p.Text); t != "" {
			if sb.Len() > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(t)
		}
		out.Segments = append(out.Segments, p.Segments...)
		out.Words = append(out.Words, p.Words...)
		if out.Language == "" {
			out.Language = p.Language
		}
	}
	out.Text = sb.String()
	return out
}
