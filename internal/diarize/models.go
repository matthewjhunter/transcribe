package diarize

import (
	"archive/tar"
	"compress/bzip2"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const (
	DefaultSegmentationFile = "sherpa-onnx-pyannote-segmentation-3-0.onnx"

	segmentationURL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
		"speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"

	// segmentationInnerSuffix is the path of the model.onnx file
	// inside the upstream tarball. It lives under a directory named
	// after the release; we look for any "*/model.onnx" entry.
	segmentationInnerSuffix = "/model.onnx"
)

// EmbeddingPreset names a speaker-embedding model from the sherpa-onnx
// model zoo. Add entries to embeddingPresets to support more.
type EmbeddingPreset string

const (
	// EmbeddingTitanetSmall is the default — ~22 MB, fast, decent
	// English discrimination.
	EmbeddingTitanetSmall EmbeddingPreset = "titanet_small"

	// EmbeddingTitanetLarge is the larger NeMo TitaNet — ~95 MB, more
	// discriminative; reduces over-clustering on noisy conversational
	// audio at the cost of a slower diarization stage.
	EmbeddingTitanetLarge EmbeddingPreset = "titanet_large"

	DefaultEmbeddingPreset = EmbeddingTitanetSmall
)

type embeddingDescriptor struct {
	Filename string
	URL      string
}

// Note the upstream release tag has a typo: "speaker-recongition-models".
var embeddingPresets = map[EmbeddingPreset]embeddingDescriptor{
	EmbeddingTitanetSmall: {
		Filename: "nemo_en_titanet_small.onnx",
		URL: "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
			"speaker-recongition-models/nemo_en_titanet_small.onnx",
	},
	EmbeddingTitanetLarge: {
		Filename: "nemo_en_titanet_large.onnx",
		URL: "https://github.com/k2-fsa/sherpa-onnx/releases/download/" +
			"speaker-recongition-models/nemo_en_titanet_large.onnx",
	},
}

// EmbeddingPresets returns the names of the known embedding presets in
// a stable order; useful for usage strings.
func EmbeddingPresets() []EmbeddingPreset {
	return []EmbeddingPreset{EmbeddingTitanetSmall, EmbeddingTitanetLarge}
}

// CacheDir returns the directory where transcribe stores its model
// files: $XDG_CACHE_HOME/transcribe/models, or $HOME/.cache/transcribe/models.
func CacheDir() (string, error) {
	root, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("diarize: locate cache dir: %w", err)
	}
	return filepath.Join(root, "transcribe", "models"), nil
}

// EnsureModels makes sure both required model files exist in CacheDir,
// downloading them from the sherpa-onnx GitHub releases when they're
// missing. Returns the absolute paths of the segmentation and embedding
// files in that order.
//
// preset selects which speaker-embedding model to fetch; see
// EmbeddingPresets. Pass "" to use DefaultEmbeddingPreset.
//
// Authenticity is not verified — these are the canonical sherpa-onnx
// release artifacts and we trust GitHub's TLS as the integrity boundary.
// If you don't trust that, supply your own paths via Config and skip
// EnsureModels.
func EnsureModels(ctx context.Context, preset EmbeddingPreset) (segPath, embPath string, err error) {
	if preset == "" {
		preset = DefaultEmbeddingPreset
	}
	desc, ok := embeddingPresets[preset]
	if !ok {
		return "", "", fmt.Errorf("diarize: unknown embedding preset %q", preset)
	}

	dir, err := CacheDir()
	if err != nil {
		return "", "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", "", fmt.Errorf("diarize: mkdir %q: %w", dir, err)
	}

	segPath = filepath.Join(dir, DefaultSegmentationFile)
	if err := ensureSegmentation(ctx, segPath); err != nil {
		return "", "", err
	}

	embPath = filepath.Join(dir, desc.Filename)
	if err := ensureFile(ctx, embPath, desc.URL); err != nil {
		return "", "", err
	}

	return segPath, embPath, nil
}

// ensureFile downloads url to dst when dst doesn't already exist.
func ensureFile(ctx context.Context, dst, url string) error {
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	tmp := dst + ".part"
	if err := download(ctx, url, tmp); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		return fmt.Errorf("diarize: rename %q -> %q: %w", tmp, dst, err)
	}
	return nil
}

// ensureSegmentation downloads the pyannote tar.bz2 if dst doesn't
// already exist, extracts the inner model.onnx, writes it to dst, and
// drops the archive.
func ensureSegmentation(ctx context.Context, dst string) error {
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	tmp := dst + ".tar.bz2"
	if err := download(ctx, segmentationURL, tmp); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	defer os.Remove(tmp)

	if err := extractSegmentationONNX(tmp, dst); err != nil {
		return err
	}
	return nil
}

func extractSegmentationONNX(archivePath, dst string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("diarize: open archive %q: %w", archivePath, err)
	}
	defer f.Close()

	tr := tar.NewReader(bzip2.NewReader(f))
	for {
		hdr, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("diarize: read tar: %w", err)
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if !strings.HasSuffix(hdr.Name, segmentationInnerSuffix) {
			continue
		}
		out, err := os.Create(dst)
		if err != nil {
			return fmt.Errorf("diarize: create %q: %w", dst, err)
		}
		if _, err := io.Copy(out, tr); err != nil {
			_ = out.Close()
			return fmt.Errorf("diarize: write %q: %w", dst, err)
		}
		return out.Close()
	}
	return fmt.Errorf("diarize: %s not found in %q", segmentationInnerSuffix, archivePath)
}

func download(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("diarize: build request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("diarize: GET %q: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("diarize: GET %q: HTTP %d", url, resp.StatusCode)
	}
	out, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("diarize: create %q: %w", dst, err)
	}
	defer out.Close()
	if _, err := io.Copy(out, resp.Body); err != nil {
		return fmt.Errorf("diarize: write %q: %w", dst, err)
	}
	return nil
}
