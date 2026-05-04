// transcribe produces a speaker-labeled transcript from an audio or video file.
//
// Pipeline: ffprobe -> ffmpeg extract canonical 16 kHz mono WAV ->
// (Whisper HTTP transcription || sherpa-onnx diarization) -> aligner ->
// rendered output.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"path/filepath"
	"runtime/debug"
	"strings"
	"syscall"

	"golang.org/x/sync/errgroup"

	"github.com/matthewjhunter/transcribe/internal/align"
	"github.com/matthewjhunter/transcribe/internal/audio"
	"github.com/matthewjhunter/transcribe/internal/diarize"
	"github.com/matthewjhunter/transcribe/internal/output"
	"github.com/matthewjhunter/transcribe/internal/whisper"
)

func main() {
	if err := run(); err != nil {
		// Errors are logged in run(); exit nonzero so shell loops notice.
		os.Exit(1)
	}
}

type config struct {
	input string

	outputPath   string
	outputFormat output.Format

	whisperURL    string
	whisperModel  string
	whisperAPIKey string
	language      string

	noDiarize        bool
	numSpeakers      int
	threshold        float64
	segmentationOnnx string
	embeddingOnnx    string

	keepTemp bool
	verbose  bool
}

func run() error {
	cfg, err := parseFlags(os.Args[1:])
	if err != nil {
		// flag.Parse already wrote a usage hint; surface a short error.
		fmt.Fprintln(os.Stderr, err)
		return err
	}

	level := slog.LevelInfo
	if cfg.verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(log)

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	if err := transcribeOne(ctx, log, cfg); err != nil {
		log.Error("transcribe failed", "err", err)
		return err
	}
	return nil
}

func parseFlags(args []string) (config, error) {
	fs := flag.NewFlagSet("transcribe", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)

	var (
		cfg     config
		format  string
		showVer bool
	)
	apiKeyDefault := os.Getenv("WHISPER_API_KEY")

	fs.StringVar(&cfg.outputPath, "output", "", "Output path. Default: <input-without-ext>.txt next to the input. Use - for stdout.")
	fs.StringVar(&cfg.outputPath, "o", "", "Shorthand for --output.")
	fs.StringVar(&format, "output-format", "tstxt", "tstxt | wxtxt | json")
	fs.StringVar(&cfg.whisperURL, "whisper-url", whisper.DefaultEndpoint, "OpenAI-compatible /v1 base URL.")
	fs.StringVar(&cfg.whisperModel, "whisper-model", whisper.DefaultModel, "Model name passed to the backend.")
	fs.StringVar(&cfg.whisperAPIKey, "whisper-api-key", apiKeyDefault, "Bearer token; defaults to $WHISPER_API_KEY.")
	fs.StringVar(&cfg.language, "language", "en", "ISO-639-1 language hint. Empty for auto-detect.")
	fs.BoolVar(&cfg.noDiarize, "no-diarize", false, "Skip speaker diarization; emit one line per segment.")
	fs.IntVar(&cfg.numSpeakers, "num-speakers", 0, "Force N speakers (0 = auto via clustering).")
	fs.Float64Var(&cfg.threshold, "speaker-threshold", 0.5, "Clustering threshold when --num-speakers=0.")
	fs.StringVar(&cfg.segmentationOnnx, "segmentation-model", "", "Path to pyannote segmentation ONNX. Empty = auto-cache.")
	fs.StringVar(&cfg.embeddingOnnx, "embedding-model", "", "Path to speaker-embedding ONNX. Empty = auto-cache.")
	fs.BoolVar(&cfg.keepTemp, "keep-temp", false, "Don't delete the extracted WAV.")
	fs.BoolVar(&cfg.verbose, "verbose", false, "Debug logging.")
	fs.BoolVar(&showVer, "version", false, "Print version and exit.")
	fs.BoolVar(&showVer, "v", false, "Shorthand for --version.")

	fs.Usage = func() {
		fmt.Fprintln(fs.Output(), "Usage: transcribe [flags] <audio-or-video-file>")
		fmt.Fprintln(fs.Output(), "")
		fs.PrintDefaults()
	}

	if err := fs.Parse(args); err != nil {
		return config{}, err
	}
	if showVer {
		fmt.Println(version())
		os.Exit(0)
	}

	if fs.NArg() != 1 {
		fs.Usage()
		return config{}, errors.New("transcribe: exactly one input file is required")
	}
	cfg.input = fs.Arg(0)

	switch strings.ToLower(format) {
	case "tstxt":
		cfg.outputFormat = output.FormatTimestampedTXT
	case "wxtxt":
		cfg.outputFormat = output.FormatWhisperXTXT
	case "json":
		cfg.outputFormat = output.FormatJSON
	default:
		return config{}, fmt.Errorf("transcribe: unknown --output-format %q (want tstxt|wxtxt|json)", format)
	}

	return cfg, nil
}

func transcribeOne(ctx context.Context, log *slog.Logger, cfg config) error {
	probe, err := audio.ProbeFile(ctx, cfg.input)
	if err != nil {
		return fmt.Errorf("probe: %w", err)
	}
	if !probe.HasAudio {
		return fmt.Errorf("input %q has no audio stream", cfg.input)
	}
	log.Info("probed input", "duration", probe.Duration, "has_video", probe.HasVideo)

	tmpDir, err := os.MkdirTemp("", "transcribe-")
	if err != nil {
		return fmt.Errorf("create temp dir: %w", err)
	}
	if cfg.keepTemp {
		log.Info("keeping temp dir", "path", tmpDir)
	} else {
		defer os.RemoveAll(tmpDir)
	}
	wavPath := filepath.Join(tmpDir, "input.wav")

	log.Info("extracting WAV", "out", wavPath)
	if err := audio.ExtractWAV(ctx, cfg.input, wavPath); err != nil {
		return fmt.Errorf("extract WAV: %w", err)
	}

	var diarizer *diarize.Diarizer
	if !cfg.noDiarize {
		segPath, embPath, err := resolveModelPaths(ctx, log, cfg)
		if err != nil {
			return err
		}
		diarizer, err = diarize.New(diarize.Config{
			SegmentationModel: segPath,
			EmbeddingModel:    embPath,
			NumSpeakers:       cfg.numSpeakers,
			Threshold:         float32(cfg.threshold),
			Debug:             cfg.verbose,
		})
		if err != nil {
			return fmt.Errorf("init diarizer: %w", err)
		}
		defer diarizer.Close()
	}

	wc := whisper.New(whisper.Config{
		Endpoint: cfg.whisperURL,
		APIKey:   cfg.whisperAPIKey,
		Model:    cfg.whisperModel,
		Language: cfg.language,
	})

	var (
		result *whisper.Result
		turns  []diarize.Turn
	)

	g, gctx := errgroup.WithContext(ctx)
	g.Go(func() error {
		log.Info("transcribing", "url", cfg.whisperURL, "model", cfg.whisperModel)
		r, err := wc.Transcribe(gctx, wavPath)
		if err != nil {
			return fmt.Errorf("whisper: %w", err)
		}
		result = r
		log.Info("transcription complete", "segments", len(r.Segments), "words", len(r.Words))
		return nil
	})
	if diarizer != nil {
		g.Go(func() error {
			log.Info("diarizing", "sample_rate", diarizer.SampleRate())
			samples, sr, err := audio.ReadFloat32(wavPath)
			if err != nil {
				return fmt.Errorf("read WAV: %w", err)
			}
			t, err := diarizer.Process(samples, sr)
			if err != nil {
				return fmt.Errorf("diarize: %w", err)
			}
			turns = t
			log.Info("diarization complete", "turns", len(t))
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	lines := buildLines(log, result, turns, cfg.noDiarize)

	w, closeFn, err := openOutput(cfg)
	if err != nil {
		return err
	}
	defer closeFn()
	if err := output.Render(lines, w, cfg.outputFormat); err != nil {
		return fmt.Errorf("render: %w", err)
	}
	return nil
}

func resolveModelPaths(ctx context.Context, log *slog.Logger, cfg config) (string, string, error) {
	if cfg.segmentationOnnx != "" && cfg.embeddingOnnx != "" {
		return cfg.segmentationOnnx, cfg.embeddingOnnx, nil
	}
	log.Info("ensuring diarization models in cache")
	seg, emb, err := diarize.EnsureModels(ctx)
	if err != nil {
		return "", "", fmt.Errorf("ensure models: %w", err)
	}
	if cfg.segmentationOnnx != "" {
		seg = cfg.segmentationOnnx
	}
	if cfg.embeddingOnnx != "" {
		emb = cfg.embeddingOnnx
	}
	return seg, emb, nil
}

func buildLines(log *slog.Logger, result *whisper.Result, turns []diarize.Turn, noDiarize bool) []align.SpeakerLine {
	if !noDiarize && len(turns) > 0 {
		if len(result.Words) > 0 {
			return align.AssignSpeakers(result.Words, turns)
		}
		log.Warn("backend returned no word timestamps; falling back to segment-level alignment")
		return align.AssignFromSegments(result.Segments, turns)
	}

	lines := make([]align.SpeakerLine, len(result.Segments))
	for i, s := range result.Segments {
		lines[i] = align.SpeakerLine{
			Start:   s.Start,
			End:     s.End,
			Speaker: 0,
			Text:    strings.TrimLeft(s.Text, " "),
		}
	}
	return lines
}

func openOutput(cfg config) (out *os.File, closeFn func(), err error) {
	dst := cfg.outputPath
	if dst == "-" {
		return os.Stdout, func() {}, nil
	}
	if dst == "" {
		ext := filepath.Ext(cfg.input)
		dst = strings.TrimSuffix(cfg.input, ext) + ".txt"
	}
	f, err := os.Create(dst)
	if err != nil {
		return nil, nil, fmt.Errorf("create %q: %w", dst, err)
	}
	return f, func() { _ = f.Close() }, nil
}

func version() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "transcribe (devel)"
}
