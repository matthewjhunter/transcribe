package diarize

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCacheDir(t *testing.T) {
	d, err := CacheDir()
	if err != nil {
		t.Fatalf("CacheDir: %v", err)
	}
	if !strings.HasSuffix(filepath.ToSlash(d), "/transcribe/models") {
		t.Errorf("CacheDir = %q, want suffix /transcribe/models", d)
	}
}

func TestEnsureFile_AlreadyExists(t *testing.T) {
	dst := filepath.Join(t.TempDir(), "exists.bin")
	if err := os.WriteFile(dst, []byte("hello"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		t.Error("server should not be hit when dst already exists")
	}))
	defer srv.Close()
	if err := ensureFile(context.Background(), dst, srv.URL); err != nil {
		t.Fatalf("ensureFile: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if string(got) != "hello" {
		t.Errorf("file rewritten unexpectedly: %q", got)
	}
}

func TestEnsureFile_Downloads(t *testing.T) {
	want := []byte("payload-bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(want)
	}))
	defer srv.Close()
	dst := filepath.Join(t.TempDir(), "out.bin")
	if err := ensureFile(context.Background(), dst, srv.URL); err != nil {
		t.Fatalf("ensureFile: %v", err)
	}
	got, _ := os.ReadFile(dst)
	if !bytes.Equal(got, want) {
		t.Errorf("contents: got %q want %q", got, want)
	}
}

func TestEnsureFile_HTTPErrorRemovesPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()
	dst := filepath.Join(t.TempDir(), "out.bin")
	err := ensureFile(context.Background(), dst, srv.URL)
	if err == nil || !strings.Contains(err.Error(), "HTTP 500") {
		t.Errorf("ensureFile error: got %v want HTTP 500", err)
	}
	if _, err := os.Stat(dst + ".part"); !errors.Is(err, os.ErrNotExist) {
		t.Error(".part file should have been removed on failure")
	}
}

// TestTarSearchPicksModelOnnx exercises the inner-tar walk logic that
// extractSegmentationONNX uses, without depending on a real bzip2
// stream (which stdlib can't produce). It's not a full end-to-end test
// but it covers the suffix-match + IO behavior.
func TestTarSearchPicksModelOnnx(t *testing.T) {
	var buf bytes.Buffer
	tw := tar.NewWriter(&buf)
	for _, f := range []struct{ name, body string }{
		{"release/README.md", "decoy"},
		{"release/model.onnx", "PAYLOAD"},
		{"release/notes.txt", "more decoy"},
	} {
		body := []byte(f.body)
		_ = tw.WriteHeader(&tar.Header{Name: f.name, Size: int64(len(body)), Mode: 0o644, Typeflag: tar.TypeReg})
		_, _ = tw.Write(body)
	}
	_ = tw.Close()

	tr := tar.NewReader(&buf)
	var got []byte
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("tar Next: %v", err)
		}
		if !strings.HasSuffix(hdr.Name, segmentationInnerSuffix) {
			continue
		}
		got, _ = io.ReadAll(tr)
	}
	if string(got) != "PAYLOAD" {
		t.Errorf("got %q want PAYLOAD", got)
	}
}
