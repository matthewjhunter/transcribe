package diarize

import "time"

// Turn is a contiguous interval attributed to a single speaker.
//
// Speaker is a 0-indexed integer assigned by the diarizer. Two turns
// with the same Speaker value belong to the same physical voice; the
// integer itself has no meaning beyond identity within one Process call.
type Turn struct {
	Start   time.Duration
	End     time.Duration
	Speaker int
}
