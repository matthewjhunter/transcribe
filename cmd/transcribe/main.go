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
	"time"

	"golang.org/x/sync/errgroup"

	"github.com/matthewjhunter/transcribe/internal/align"
	"github.com/matthewjhunter/transcribe/internal/audio"
	"github.com/matthewjhunter/transcribe/internal/diarize"
	"github.com/matthewjhunter/transcribe/internal/output"
	"github.com/matthewjhunter/transcribe/internal/vad"
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

	whisperURL         string
	whisperModel       string
	whisperAPIKey      string
	whisperConcurrency int
	language           string

	noVAD         bool
	vadMinSilence float64
	vadMaxChunk   float64
	vadMinChunk   float64
	vadModel      string

	noExtractAudio bool

	noDiarize        bool
	numSpeakers      int
	minSpeechDur     float64
	minSilenceDur    float64
	diarizeThreads   int
	diarizeProvider  string
	embeddingPreset  string
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
	fs.IntVar(&cfg.whisperConcurrency, "whisper-concurrency", 1, "Parallel transcription requests when VAD chunking is on.")
	fs.StringVar(&cfg.language, "language", "en", "ISO-639-1 language hint. Empty for auto-detect.")
	fs.BoolVar(&cfg.noVAD, "no-vad", false, "Disable VAD pre-chunking; submit the full file as one request.")
	fs.Float64Var(&cfg.vadMinSilence, "vad-min-silence", 0.5, "Merge VAD speech regions separated by < N seconds of silence.")
	fs.Float64Var(&cfg.vadMaxChunk, "vad-max-chunk", 28, "Hard cap on chunk length sent to the backend, in seconds.")
	fs.Float64Var(&cfg.vadMinChunk, "vad-min-chunk", 1, "Drop chunks shorter than N seconds after merging.")
	fs.StringVar(&cfg.vadModel, "vad-model", "", "Path to silero_vad.onnx. Empty = auto-cache.")
	fs.BoolVar(&cfg.noExtractAudio, "no-extract-audio", false, "Don't write a sibling lossless audio-stream copy next to the transcript.")
	fs.BoolVar(&cfg.noDiarize, "no-diarize", false, "Skip speaker diarization; emit one line per segment.")
	fs.IntVar(&cfg.numSpeakers, "num-speakers", 0, "Required (unless --no-diarize). Number of distinct speakers in the recording.")
	fs.Float64Var(&cfg.minSpeechDur, "min-speech-duration", 0, "Drop speech segments shorter than N seconds. 0 = sherpa default.")
	fs.Float64Var(&cfg.minSilenceDur, "min-silence-duration", 0, "Merge speech segments separated by < N seconds of silence. 0 = sherpa default.")
	fs.IntVar(&cfg.diarizeThreads, "diarize-threads", 0, "Threads for sherpa segmentation/embedding stages. 0 = NumCPU.")
	fs.StringVar(&cfg.diarizeProvider, "diarize-provider", "", "ONNX execution provider for diarization (cpu|cuda|...). Empty = cpu.")
	fs.StringVar(&cfg.embeddingPreset, "embedding-preset", string(diarize.DefaultEmbeddingPreset),
		"Speaker-embedding preset: "+joinPresets()+". Ignored when --embedding-model is set.")
	fs.StringVar(&cfg.segmentationOnnx, "segmentation-model", "", "Path to pyannote segmentation ONNX. Empty = auto-cache.")
	fs.StringVar(&cfg.embeddingOnnx, "embedding-model", "", "Path to speaker-embedding ONNX. Empty = auto-cache (see --embedding-preset).")
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

	if !cfg.noDiarize && cfg.numSpeakers <= 0 {
		fs.Usage()
		return config{}, errors.New("transcribe: --num-speakers must be set to a positive integer (or pass --no-diarize)")
	}

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
	log.Info("probed input", "duration", probe.Duration, "has_video", probe.HasVideo, "audio_codec", probe.AudioCodec)

	if !cfg.noExtractAudio && probe.HasVideo {
		if err := extractAudioSibling(ctx, log, cfg, probe); err != nil {
			return err
		}
	}

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

	samples, sampleRate, err := audio.ReadFloat32(wavPath)
	if err != nil {
		return fmt.Errorf("read WAV: %w", err)
	}

	var chunks []vad.Chunk
	if !cfg.noVAD {
		chunks, err = planVADChunks(ctx, log, cfg, samples, sampleRate)
		if err != nil {
			return err
		}
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
			MinDurationOn:     float32(cfg.minSpeechDur),
			MinDurationOff:    float32(cfg.minSilenceDur),
			NumThreads:        cfg.diarizeThreads,
			Provider:          cfg.diarizeProvider,
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
		var (
			r   *whisper.Result
			err error
		)
		if cfg.noVAD {
			log.Info("transcribing whole file", "url", cfg.whisperURL, "model", cfg.whisperModel)
			r, err = wc.Transcribe(gctx, wavPath)
		} else {
			log.Info("transcribing chunks", "url", cfg.whisperURL, "model", cfg.whisperModel,
				"chunks", len(chunks), "concurrency", cfg.whisperConcurrency)
			r, err = wc.TranscribeChunks(gctx, samples, sampleRate, chunks, cfg.whisperConcurrency)
		}
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
			t, err := diarizer.Process(samples, sampleRate)
			if err != nil {
				return fmt.Errorf("diarize: %w", err)
			}
			turns = t
			log.Info("diarization complete", "turns", len(t), "raw_speakers", distinctSpeakers(t))
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

// extractAudioSibling writes a lossless stream-copy of the input's
// audio next to wherever the transcript will land. Skips silently when
// the destination already exists (idempotent re-runs) or when the
// destination resolves to nothing useful (stdout output).
func extractAudioSibling(ctx context.Context, log *slog.Logger, cfg config, probe audio.Probe) error {
	dst := siblingAudioPath(cfg, probe.AudioCodec)
	if dst == "" {
		return nil
	}
	if _, err := os.Stat(dst); err == nil {
		log.Info("audio sibling already exists; skipping", "path", dst)
		return nil
	}
	log.Info("extracting audio stream", "out", dst, "codec", probe.AudioCodec)
	if err := audio.ExtractAudioStream(ctx, cfg.input, dst); err != nil {
		return fmt.Errorf("extract audio stream: %w", err)
	}
	return nil
}

// siblingAudioPath derives the lossless-audio sibling file path from
// the transcript path that openOutput would pick: the same directory,
// the same basename, with a codec-derived extension. Returns "" when
// no path makes sense (stdout output, missing codec, or no input).
func siblingAudioPath(cfg config, codec string) string {
	if codec == "" || cfg.outputPath == "-" {
		return ""
	}
	ext := audio.AudioStreamExt(codec)
	base := cfg.outputPath
	if base == "" {
		base = strings.TrimSuffix(cfg.input, filepath.Ext(cfg.input))
	} else {
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	if base == "" {
		return ""
	}
	return base + "." + ext
}

func planVADChunks(ctx context.Context, log *slog.Logger, cfg config, samples []float32, sampleRate int) ([]vad.Chunk, error) {
	modelPath := cfg.vadModel
	if modelPath == "" {
		log.Info("ensuring VAD model in cache")
		p, err := vad.EnsureModel(ctx)
		if err != nil {
			return nil, fmt.Errorf("ensure vad model: %w", err)
		}
		modelPath = p
	}

	maxChunk := time.Duration(cfg.vadMaxChunk * float64(time.Second))
	det, err := vad.New(vad.Config{
		Model:              modelPath,
		SampleRate:         sampleRate,
		MinSilenceDuration: float32(cfg.vadMinSilence),
		MaxSpeechDuration:  float32(cfg.vadMaxChunk),
		NumThreads:         cfg.diarizeThreads,
		Provider:           cfg.diarizeProvider,
		Debug:              cfg.verbose,
	})
	if err != nil {
		return nil, fmt.Errorf("init vad: %w", err)
	}
	defer det.Close()

	log.Info("running VAD", "sample_rate", sampleRate)
	segs, err := det.Detect(samples, sampleRate)
	if err != nil {
		return nil, fmt.Errorf("vad: %w", err)
	}
	chunks := vad.Plan(segs, vad.PlanOptions{
		MinSilence: time.Duration(cfg.vadMinSilence * float64(time.Second)),
		MaxChunk:   maxChunk,
		MinChunk:   time.Duration(cfg.vadMinChunk * float64(time.Second)),
	})
	log.Info("vad plan complete", "raw_segments", len(segs), "chunks", len(chunks))
	if log.Enabled(ctx, slog.LevelDebug) {
		for i, ch := range chunks {
			log.Debug("vad chunk", "i", i, "start", ch.Start, "end", ch.End, "span", ch.End-ch.Start)
		}
	}
	return chunks, nil
}

func resolveModelPaths(ctx context.Context, log *slog.Logger, cfg config) (string, string, error) {
	if cfg.segmentationOnnx != "" && cfg.embeddingOnnx != "" {
		return cfg.segmentationOnnx, cfg.embeddingOnnx, nil
	}
	preset := diarize.EmbeddingPreset(cfg.embeddingPreset)
	log.Info("ensuring diarization models in cache", "embedding_preset", preset)
	seg, emb, err := diarize.EnsureModels(ctx, preset)
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

func joinPresets() string {
	names := diarize.EmbeddingPresets()
	parts := make([]string, len(names))
	for i, n := range names {
		parts[i] = string(n)
	}
	return strings.Join(parts, " | ")
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

func distinctSpeakers(turns []diarize.Turn) int {
	seen := make(map[int]struct{}, len(turns))
	for _, t := range turns {
		seen[t.Speaker] = struct{}{}
	}
	return len(seen)
}

func version() string {
	if info, ok := debug.ReadBuildInfo(); ok && info.Main.Version != "" && info.Main.Version != "(devel)" {
		return info.Main.Version
	}
	return "transcribe (devel)"
}
