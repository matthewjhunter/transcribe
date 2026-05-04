// transcribe produces a speaker-labeled transcript from an audio or video file.
//
// See ../../CLAUDE.md for the design plan. v0.0.0 — entrypoint stub only.
package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: transcribe <audio-or-video-file>")
		os.Exit(2)
	}
	fmt.Fprintln(os.Stderr, "transcribe: not yet implemented")
	os.Exit(1)
}
