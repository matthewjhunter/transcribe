package vad

import (
	"reflect"
	"testing"
	"time"
)

func seg(startMs, endMs int) Segment {
	return Segment{
		Start: time.Duration(startMs) * time.Millisecond,
		End:   time.Duration(endMs) * time.Millisecond,
	}
}

func chunk(startMs, endMs int) Chunk {
	return Chunk{
		Start: time.Duration(startMs) * time.Millisecond,
		End:   time.Duration(endMs) * time.Millisecond,
	}
}

func TestPlan_Empty(t *testing.T) {
	got := Plan(nil, PlanOptions{})
	if got != nil {
		t.Errorf("Plan(nil) = %v, want nil", got)
	}
}

func TestPlan_AppliesDefaults(t *testing.T) {
	// A single 5s segment should pass through with default options
	// (MinChunk=1s, MaxChunk=28s, MinSilence=500ms).
	in := []Segment{seg(0, 5000)}
	got := Plan(in, PlanOptions{})
	want := []Chunk{chunk(0, 5000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(...) = %v, want %v", got, want)
	}
}

func TestPlan_DropsShort(t *testing.T) {
	// A 0.5s segment is below default MinChunk=1s and should be dropped.
	in := []Segment{seg(0, 500)}
	got := Plan(in, PlanOptions{})
	if len(got) != 0 {
		t.Errorf("Plan(short) = %v, want empty", got)
	}
}

func TestPlan_MergesSmallGap(t *testing.T) {
	// Two adjacent segments with a 200ms gap (< 500ms default) merge.
	in := []Segment{seg(0, 5000), seg(5200, 10000)}
	got := Plan(in, PlanOptions{})
	want := []Chunk{chunk(0, 10000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(...) = %v, want %v", got, want)
	}
}

func TestPlan_KeepsLargeGap(t *testing.T) {
	// Two segments with a 2s gap (> 500ms default) stay separate.
	in := []Segment{seg(0, 5000), seg(7000, 12000)}
	got := Plan(in, PlanOptions{})
	want := []Chunk{chunk(0, 5000), chunk(7000, 12000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(...) = %v, want %v", got, want)
	}
}

func TestPlan_RefusesMergeOverMaxChunk(t *testing.T) {
	// Two segments with a small gap whose merged span would exceed
	// MaxChunk=10s stay separate.
	in := []Segment{seg(0, 6000), seg(6200, 11000)}
	got := Plan(in, PlanOptions{MaxChunk: 10 * time.Second})
	want := []Chunk{chunk(0, 6000), chunk(6200, 11000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(...) = %v, want %v", got, want)
	}
}

func TestPlan_HardSplitsOversize(t *testing.T) {
	// A single 30s segment with MaxChunk=10s splits into 3 equal pieces.
	in := []Segment{seg(0, 30000)}
	got := Plan(in, PlanOptions{MaxChunk: 10 * time.Second})
	want := []Chunk{
		chunk(0, 10000),
		chunk(10000, 20000),
		chunk(20000, 30000),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(oversize) = %v, want %v", got, want)
	}
}

func TestPlan_HardSplitsUnevenOversize(t *testing.T) {
	// A 25s segment with MaxChunk=10s splits into 3 pieces. Ceil(25/10)=3
	// so each piece is ~8.33s; the last piece carries the remainder
	// exactly to End to avoid float drift.
	in := []Segment{seg(0, 25000)}
	got := Plan(in, PlanOptions{MaxChunk: 10 * time.Second})
	if len(got) != 3 {
		t.Fatalf("Plan(oversize uneven) = %v, want 3 pieces", got)
	}
	if got[0].Start != 0 || got[2].End != 25000*time.Millisecond {
		t.Errorf("piece boundaries wrong: %v", got)
	}
	for _, c := range got {
		if c.End-c.Start > 10*time.Second {
			t.Errorf("piece %v exceeds MaxChunk", c)
		}
	}
}

func TestPlan_DropsTinyAfterMerge(t *testing.T) {
	// A 200ms back-channel between two long talks with 1s gaps either
	// side: gaps exceed MinSilence so the tiny piece doesn't merge,
	// then it's dropped for being below MinChunk.
	in := []Segment{seg(0, 5000), seg(6000, 6200), seg(7200, 12000)}
	got := Plan(in, PlanOptions{})
	want := []Chunk{chunk(0, 5000), chunk(7200, 12000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(tiny middle) = %v, want %v", got, want)
	}
}

func TestPlan_MergesTinyIntoNeighbor(t *testing.T) {
	// A 200ms back-channel with small gaps either side merges into the
	// previous chunk on the first pass; the result is one continuous chunk.
	in := []Segment{seg(0, 5000), seg(5300, 5500), seg(5800, 10000)}
	got := Plan(in, PlanOptions{})
	want := []Chunk{chunk(0, 10000)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(tiny merged) = %v, want %v", got, want)
	}
}

func TestPlan_MixedScenario(t *testing.T) {
	// One long monologue, a clean turn boundary, two close-together
	// utterances, a tiny stranded back-channel, and a final long talk.
	in := []Segment{
		seg(0, 8000),
		seg(15000, 20000),
		seg(20300, 25000),
		seg(40000, 40400),
		seg(60000, 70000),
	}
	got := Plan(in, PlanOptions{})
	want := []Chunk{
		chunk(0, 8000),
		chunk(15000, 25000),
		chunk(60000, 70000),
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Plan(mixed) = %v, want %v", got, want)
	}
}

func TestPlan_HardSplitThenMerge(t *testing.T) {
	// A 20s segment hard-splits at MaxChunk=8s into pieces of ~6.67s.
	// They are contiguous (gap=0) so they would merge -- but their
	// merged span equals the original 20s which exceeds MaxChunk. The
	// merge guard prevents that, so we keep three split pieces.
	in := []Segment{seg(0, 20000)}
	got := Plan(in, PlanOptions{MaxChunk: 8 * time.Second})
	if len(got) != 3 {
		t.Fatalf("Plan(split-then-merge guard) = %v, want 3", got)
	}
	for _, c := range got {
		if c.End-c.Start > 8*time.Second {
			t.Errorf("piece %v exceeds MaxChunk after pass-through merge", c)
		}
	}
}
