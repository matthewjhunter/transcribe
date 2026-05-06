package voicelabel

import (
	"math"
	"testing"
	"time"

	"github.com/matthewjhunter/transcribe/internal/diarize"
)

func mixSineIntoBuffer(buf []float32, freq float64, sampleRate, startSample int, duration time.Duration) {
	n := int(float64(sampleRate) * duration.Seconds())
	for i := range n {
		idx := startSample + i
		if idx >= len(buf) {
			break
		}
		buf[idx] = float32(0.5 * math.Sin(2*math.Pi*freq*float64(i)/float64(sampleRate)))
	}
}

// TestClassifyClusters_AdultMaleVsFemale builds a 6 s buffer with two
// diarization clusters: cluster 0 is a 130 Hz tone (adult-male typical),
// cluster 1 is a 220 Hz tone (adult-female typical). Verifies the labels
// come out M and F.
func TestClassifyClusters_AdultMaleVsFemale(t *testing.T) {
	const sr = 16000
	buf := make([]float32, sr*6)
	mixSineIntoBuffer(buf, 130, sr, 0, 3*time.Second)
	mixSineIntoBuffer(buf, 220, sr, 3*sr, 3*time.Second)

	turns := []diarize.Turn{
		{Start: 0, End: 3 * time.Second, Speaker: 0},
		{Start: 3 * time.Second, End: 6 * time.Second, Speaker: 1},
	}

	stats := ClassifyClusters(buf, sr, turns, ClassifyOptions{})
	if len(stats) != 2 {
		t.Fatalf("got %d clusters, want 2", len(stats))
	}
	if stats[0].Label != LabelMale {
		t.Errorf("cluster 0 (130 Hz): label = %q, want M (median F0 = %.1f)", stats[0].Label, stats[0].MedianF0Hz)
	}
	if stats[1].Label != LabelFemale {
		t.Errorf("cluster 1 (220 Hz): label = %q, want F (median F0 = %.1f)", stats[1].Label, stats[1].MedianF0Hz)
	}
}

// TestClassifyClusters_CrossoverIsUnknown verifies a voice in the
// 155–180 Hz crossover zone produces the "?" label rather than
// committing to M or F.
func TestClassifyClusters_CrossoverIsUnknown(t *testing.T) {
	const sr = 16000
	buf := make([]float32, sr*3)
	mixSineIntoBuffer(buf, 168, sr, 0, 3*time.Second) // crossover

	turns := []diarize.Turn{{Start: 0, End: 3 * time.Second, Speaker: 7}}

	stats := ClassifyClusters(buf, sr, turns, ClassifyOptions{})
	if got := stats[7].Label; got != LabelUnknown {
		t.Errorf("crossover voice: label = %q (median %.1f), want ?", got, stats[7].MedianF0Hz)
	}
}

// TestClassifyClusters_SilentClusterIsUnknown verifies a cluster whose
// audio is silent (no voiced frames) gets LabelUnknown rather than a
// spurious M/F.
func TestClassifyClusters_SilentClusterIsUnknown(t *testing.T) {
	const sr = 16000
	buf := make([]float32, sr*3) // all zeros

	turns := []diarize.Turn{{Start: 0, End: 3 * time.Second, Speaker: 0}}

	stats := ClassifyClusters(buf, sr, turns, ClassifyOptions{})
	if got := stats[0].Label; got != LabelUnknown {
		t.Errorf("silent cluster: label = %q, want ?", got)
	}
}

// TestClassifyClusters_RespectsMaxClusterAudio verifies the cap on
// per-cluster audio: with MaxClusterAudio=1s, a 5s cluster only feeds
// the first 1s of its audio into YIN.
func TestClassifyClusters_RespectsMaxClusterAudio(t *testing.T) {
	const sr = 16000
	buf := make([]float32, sr*5)
	// First 1 s: 130 Hz (M). Remaining 4 s: 220 Hz (F).
	mixSineIntoBuffer(buf, 130, sr, 0, time.Second)
	mixSineIntoBuffer(buf, 220, sr, sr, 4*time.Second)

	turns := []diarize.Turn{{Start: 0, End: 5 * time.Second, Speaker: 0}}

	opts := ClassifyOptions{MaxClusterAudio: time.Second}
	stats := ClassifyClusters(buf, sr, turns, opts)
	if got := stats[0].Label; got != LabelMale {
		t.Errorf("with 1 s cap, label = %q (median %.1f), want M from first second", got, stats[0].MedianF0Hz)
	}
}

// TestClassifyClusters_TooFewFramesIsUnknown verifies a cluster with
// almost no audio produces LabelUnknown rather than a confident guess.
func TestClassifyClusters_TooFewFramesIsUnknown(t *testing.T) {
	const sr = 16000
	buf := sineWave(130, sr, 0.05) // 50 ms — not enough voiced frames

	turns := []diarize.Turn{{Start: 0, End: 50 * time.Millisecond, Speaker: 3}}

	stats := ClassifyClusters(buf, sr, turns, ClassifyOptions{})
	if got := stats[3].Label; got != LabelUnknown {
		t.Errorf("brief cluster: label = %q, want ?", got)
	}
}

// TestClassifyClusters_EmptyInputs verifies safe handling of nil/empty
// inputs without panic.
func TestClassifyClusters_EmptyInputs(t *testing.T) {
	got := ClassifyClusters(nil, 16000, nil, ClassifyOptions{})
	if len(got) != 0 {
		t.Errorf("nil inputs: got %d entries, want 0", len(got))
	}
	got = ClassifyClusters([]float32{0, 0, 0}, 16000, nil, ClassifyOptions{})
	if len(got) != 0 {
		t.Errorf("no turns: got %d entries, want 0", len(got))
	}
	got = ClassifyClusters(nil, 16000, []diarize.Turn{{Speaker: 0, End: time.Second}}, ClassifyOptions{})
	if len(got) != 0 {
		t.Errorf("nil samples: got %d entries, want 0", len(got))
	}
}
