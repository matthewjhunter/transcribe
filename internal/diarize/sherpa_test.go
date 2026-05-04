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
