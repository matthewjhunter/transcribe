package diarize

import (
	"reflect"
	"testing"
	"time"
)

func d(ms int) time.Duration { return time.Duration(ms) * time.Millisecond }

func TestMergeToTarget_NoOpWhenWithinTolerance(t *testing.T) {
	in := []Turn{
		{Start: d(0), End: d(1000), Speaker: 7},
		{Start: d(1000), End: d(2000), Speaker: 3},
	}
	got := MergeToTarget(in, 4, 2)
	// Within tolerance (2 distinct <= 6); only renumbering should happen.
	want := []Turn{
		{Start: d(0), End: d(1000), Speaker: 0},
		{Start: d(1000), End: d(2000), Speaker: 1},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %+v want %+v", got, want)
	}
}

func TestMergeToTarget_DisabledByZeroTarget(t *testing.T) {
	in := []Turn{
		{Start: d(0), End: d(1000), Speaker: 7},
		{Start: d(1000), End: d(2000), Speaker: 3},
	}
	got := MergeToTarget(in, 0, 2)
	if !reflect.DeepEqual(got, in) {
		t.Errorf("expected pass-through, got %+v", got)
	}
}

func TestMergeToTarget_DropsLowestDuration(t *testing.T) {
	// Three "main" speakers and three fragment speakers. Target 2,
	// tolerance 1 → keep top 3, fold the rest into temporal neighbors.
	in := []Turn{
		{Start: d(0), End: d(5000), Speaker: 1},      // 5s
		{Start: d(5000), End: d(5050), Speaker: 9},   // 50ms fragment
		{Start: d(5050), End: d(10000), Speaker: 2},  // 5s
		{Start: d(10000), End: d(10100), Speaker: 8}, // 100ms fragment
		{Start: d(10100), End: d(15000), Speaker: 3}, // 5s
		{Start: d(15000), End: d(15050), Speaker: 7}, // 50ms fragment
		{Start: d(15050), End: d(20000), Speaker: 1}, // 5s (back to spkr 1)
	}
	got := MergeToTarget(in, 2, 1) // cap = 3
	// Speakers kept: 1, 2, 3 (largest durations). Speakers 7, 8, 9
	// reassigned to temporal neighbor.
	// After renumber-by-first-appearance: 1 -> 0, 2 -> 1, 3 -> 2.

	speakers := map[int]bool{}
	for _, t := range got {
		speakers[t.Speaker] = true
	}
	if len(speakers) != 3 {
		t.Errorf("speaker count: got %d want 3 (got speakers %v)", len(speakers), speakers)
	}
	if got[0].Speaker != 0 {
		t.Errorf("first turn speaker: got %d want 0", got[0].Speaker)
	}
	// The 50ms fragment at t=5000ms should be reassigned to either
	// spkr 1 (turn 0, center 2500) or spkr 2 (turn 2, center 7525).
	// |5025 - 2500| = 2525 vs |5025 - 7525| = 2500 → spkr 2 (now id 1).
	if got[1].Speaker != 1 {
		t.Errorf("fragment[1] reassigned to %d; expected speaker 1 (orig spkr 2)", got[1].Speaker)
	}
}

func TestMergeToTarget_NoChangeWhenAlreadyAtCap(t *testing.T) {
	in := []Turn{
		{Start: d(0), End: d(1000), Speaker: 0},
		{Start: d(1000), End: d(2000), Speaker: 1},
		{Start: d(2000), End: d(3000), Speaker: 2},
		{Start: d(3000), End: d(4000), Speaker: 3},
	}
	got := MergeToTarget(in, 2, 2) // cap = 4 == distinct
	// Already 4 distinct speakers; should pass through with renumbering.
	if len(got) != 4 {
		t.Fatalf("len: got %d want 4", len(got))
	}
	for i := range got {
		if got[i].Speaker != i {
			t.Errorf("speaker[%d] = %d; want renumbered to %d", i, got[i].Speaker, i)
		}
	}
}

func TestMergeToTarget_RenumbersByFirstAppearance(t *testing.T) {
	// Distinct count 3, cap 4 → no merge, just renumber.
	in := []Turn{
		{Start: d(0), End: d(100), Speaker: 9},
		{Start: d(100), End: d(200), Speaker: 9},
		{Start: d(200), End: d(300), Speaker: 5},
		{Start: d(300), End: d(400), Speaker: 1},
	}
	got := MergeToTarget(in, 4, 2)
	want := []int{0, 0, 1, 2} // 9 first → 0; 5 second → 1; 1 third → 2.
	for i, turn := range got {
		if turn.Speaker != want[i] {
			t.Errorf("turn[%d]: got %d want %d", i, turn.Speaker, want[i])
		}
	}
}

func TestMergeToTarget_EmptyTurns(t *testing.T) {
	if got := MergeToTarget(nil, 4, 2); got != nil {
		t.Errorf("nil input should pass through; got %+v", got)
	}
	if got := MergeToTarget([]Turn{}, 4, 2); len(got) != 0 {
		t.Errorf("empty input should pass through; got %+v", got)
	}
}

func TestMergeToTarget_NegativeToleranceClamped(t *testing.T) {
	in := []Turn{
		{Start: d(0), End: d(5000), Speaker: 1},
		{Start: d(5000), End: d(100), Speaker: 2}, // small
		{Start: d(5100), End: d(10000), Speaker: 3},
	}
	got := MergeToTarget(in, 2, -5) // negative → 0; cap = 2
	speakers := map[int]bool{}
	for _, t := range got {
		speakers[t.Speaker] = true
	}
	if len(speakers) != 2 {
		t.Errorf("expected 2 speakers, got %d", len(speakers))
	}
}
