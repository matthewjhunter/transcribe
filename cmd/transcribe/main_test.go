package main

import "testing"

func TestSiblingAudioPath(t *testing.T) {
	cases := []struct {
		name  string
		cfg   config
		codec string
		want  string
	}{
		{
			name:  "default output, video input",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "aac",
			want:  "/home/m/clip.m4a",
		},
		{
			name:  "explicit output redirects sibling too",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "/tmp/foo.txt"},
			codec: "aac",
			want:  "/tmp/foo.m4a",
		},
		{
			name:  "explicit output without extension",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "/tmp/transcript"},
			codec: "opus",
			want:  "/tmp/transcript.opus",
		},
		{
			name:  "stdout output skips sibling",
			cfg:   config{input: "/home/m/clip.mkv", outputPath: "-"},
			codec: "aac",
			want:  "",
		},
		{
			name:  "empty codec skips sibling",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "",
			want:  "",
		},
		{
			name:  "unknown codec falls back to codec-name extension",
			cfg:   config{input: "/home/m/clip.mkv"},
			codec: "weird",
			want:  "/home/m/clip.weird",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := siblingAudioPath(tc.cfg, tc.codec)
			if got != tc.want {
				t.Errorf("siblingAudioPath() = %q, want %q", got, tc.want)
			}
		})
	}
}
