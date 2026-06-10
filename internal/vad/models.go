package vad

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	// DefaultModelFile is the on-disk filename for the Silero VAD ONNX
	// inside CacheDir.
	DefaultModelFile = "silero_vad.onnx"

	// modelURL is the canonical sherpa-onnx Silero VAD release artifact.
	modelURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
		"asr-models/silero_vad.onnx"
)

// CacheDir returns the directory where transcribe stores its model
// files: $XDG_CACHE_HOME/transcribe/models, or $HOME/.cache/transcribe/models.
//
// This intentionally matches diarize.CacheDir so all model files live
// flat in one location.
func CacheDir() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("vad: locate cache dir: %w", err)
	}
	return filepath.Join(root, "transcribe", "models"), nil
}

// EnsureModel makes sure silero_vad.onnx exists in CacheDir, downloading
// it from the sherpa-onnx GitHub releases when it's missing. Returns
// the absolute path of the model file.
//
// Authenticity is not verified -- this is the canonical sherpa-onnx
// release artifact and we trust GitHub's TLS as the integrity boundary.
func EnsureModel(ctx context.Context) (string, error) {
	dir, err := CacheDir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", fmt.Errorf("vad: mkdir %q: %w", dir, err)
	}
	dst := filepath.Join(dir, DefaultModelFile)
	if _, err := os.Stat(dst); err == nil {
		return dst, nil
	}
	tmp := dst + ".part"
	if err := download(ctx, modelURL, tmp); err != nil {
		_ = os.Remove(tmp)
		return "", err
	}
	if err := os.Rename(tmp, dst); err != nil {
		return "", fmt.Errorf("vad: rename %q -> %q: %w", tmp, dst, err)
	}
	return dst, nil
}

func download(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("vad: build request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("vad: GET %q: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("vad: GET %q: HTTP %d", url, resp.StatusCode)
	}
	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("vad: create %q: %w", dst, err)
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		return fmt.Errorf("vad: write %q: %w", dst, err)
	}
	return nil
}
