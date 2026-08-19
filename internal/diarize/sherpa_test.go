package diarize

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestNew_MissingSegmentation(t *testing.T) {
	_, err := New(Config{EmbeddingModel: "anywhere.onnx"})
	if err == nil || !strings.Contains(err.Error(), "SegmentationModel") {
		t.Errorf("got %v, want SegmentationModel-required error", err)
	}
}

func TestNew_MissingEmbedding(t *testing.T) {
	_, err := New(Config{SegmentationModel: "anywhere.onnx"})
	if err == nil || !strings.Contains(err.Error(), "EmbeddingModel") {
		t.Errorf("got %v, want EmbeddingModel-required error", err)
	}
}

func TestNew_FileMissing(t *testing.T) {
	dir := t.TempDir()
	_, err := New(Config{
		SegmentationModel: filepath.Join(dir, "nope-seg.onnx"),
		EmbeddingModel:    filepath.Join(dir, "nope-emb.onnx"),
	})
	if err == nil || !strings.Contains(err.Error(), "segmentation model") {
		t.Errorf("got %v, want segmentation-model-not-found error", err)
	}
}

// The default threadpool is capped: handing sherpa every core on a many-core
// machine is measurably slower than eight threads, so the cap is the point.
func TestDefaultThreads(t *testing.T) {
	cases := []struct{ numCPU, want int }{
		{0, 1},   // degenerate input still yields a usable pool
		{1, 1},   // small machines get every core
		{4, 4},   //
		{8, 8},   // exactly at the cap
		{32, 8},  // a 32-thread box is capped, not saturated
		{128, 8}, //
	}
	for _, c := range cases {
		if got := defaultThreads(c.numCPU); got != c.want {
			t.Errorf("defaultThreads(%d) = %d, want %d", c.numCPU, got, c.want)
		}
	}
}
