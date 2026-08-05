#!/usr/bin/env bash
# Prepare a statically-linked build of sherpa-onnx for linux/amd64.
#
# The upstream Go binding (k2-fsa/sherpa-onnx-go-linux) ships shared
# libraries and bakes an rpath into the Go module cache, so the installed
# binary breaks every time the module cache is cleaned. This script builds
# a local fork of the binding that links sherpa-onnx and onnxruntime
# statically from the official static-lib release archives.
#
# It produces:
#   build/sherpa-onnx-go-linux-static/  fork of the binding with static libs
#   go.static.mod / go.static.sum       module files with the replace directive
#
# Build with: go build -modfile=go.static.mod ./cmd/transcribe
# (or just run: task build:static)
#
# When bumping sherpa-onnx-go in go.mod, update BOTH pins below to the matching
# linux-x64-static-lib release tarball. The version pin exists so that a bump
# which forgets the checksum fails with a message saying so, rather than with a
# bare checksum mismatch that looks like a corrupt download or a tampered file.
# (That is exactly how this broke: dependabot walked go.mod from v1.13.0 to
# v1.13.4 and left the checksum behind.)
#
# The published digest for a release asset is visible without downloading it:
#   curl -s https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/vX.Y.Z \
#     | jq -r '.assets[] | select(.name|endswith("linux-x64-static-lib.tar.bz2")) | .digest'
set -euo pipefail

SHERPA_PINNED_VERSION="v1.13.4"
SHERPA_SHA256="98b0e31996426f6e78244dbce1955548f2c64e8f01c4be75b85af7cdaa2e8d5c"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ARCH="$(uname -m)"
if [[ "$(uname -s)" != "Linux" || "$ARCH" != "x86_64" ]]; then
    echo "prepare-static.sh only supports linux/amd64 (got $(uname -s)/$ARCH)" >&2
    exit 1
fi

# Keep the static libs pinned to the same version go.mod uses for the binding.
SHERPA_VERSION="$(go list -m -f '{{.Version}}' github.com/k2-fsa/sherpa-onnx-go-linux)"
if [[ "$SHERPA_VERSION" != "$SHERPA_PINNED_VERSION" ]]; then
    cat >&2 <<EOF
go.mod has github.com/k2-fsa/sherpa-onnx-go-linux at $SHERPA_VERSION, but
scripts/prepare-static.sh pins the static libs at $SHERPA_PINNED_VERSION.

Update SHERPA_PINNED_VERSION and SHERPA_SHA256 together at the top of that
script. The upstream digest for the new release:

  curl -s https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/$SHERPA_VERSION \\
    | jq -r '.assets[] | select(.name|endswith("linux-x64-static-lib.tar.bz2")) | .digest'
EOF
    exit 1
fi
TARBALL="sherpa-onnx-${SHERPA_VERSION}-linux-x64-static-lib.tar.bz2"
URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${TARBALL}"

BUILD_DIR="$REPO_ROOT/build"
FORK_DIR="$BUILD_DIR/sherpa-onnx-go-linux-static"
LIB_SUBDIR="lib/x86_64-unknown-linux-gnu"
mkdir -p "$BUILD_DIR"

# 1. Fetch and verify the static-lib release archive. Download to a temp
#    name and rename only after the checksum passes, so a partial or
#    tampered file never sits at the verified path.
if ! echo "$SHERPA_SHA256  $BUILD_DIR/$TARBALL" | sha256sum --check --status >/dev/null 2>&1; then
    echo "Downloading $URL"
    curl --fail --location --silent --show-error --proto '=https' --tlsv1.2 \
        -o "$BUILD_DIR/$TARBALL.tmp" "$URL"
    echo "$SHERPA_SHA256  $BUILD_DIR/$TARBALL.tmp" | sha256sum --check --quiet
    mv "$BUILD_DIR/$TARBALL.tmp" "$BUILD_DIR/$TARBALL"
fi

# 2. Fork the binding source out of the module cache.
MOD_DIR="$(go list -m -f '{{.Dir}}' github.com/k2-fsa/sherpa-onnx-go-linux)"
rm -rf "$FORK_DIR"
mkdir -p "$FORK_DIR"
# Module cache is read-only; copy the Go sources and metadata, skip the .so libs.
find "$MOD_DIR" -maxdepth 1 -type f -exec cp {} "$FORK_DIR/" \;
chmod -R u+w "$FORK_DIR"

# 3. Unpack the static archives into the lib path the cgo directives expect.
mkdir -p "$FORK_DIR/$LIB_SUBDIR"
tar -xjf "$BUILD_DIR/$TARBALL" -C "$FORK_DIR/$LIB_SUBDIR" --strip-components=2 \
    "sherpa-onnx-${SHERPA_VERSION}-linux-x64-static-lib/lib"
if [[ ! -f "$FORK_DIR/$LIB_SUBDIR/libsherpa-onnx-c-api.a" ]]; then
    echo "extracted archive is missing libsherpa-onnx-c-api.a; release layout changed?" >&2
    exit 1
fi

# 4. Replace the dynamic link flags with a static archive group.
#    --start-group/--end-group lets ld resolve the inter-archive references
#    without hand-ordering a dozen libraries.
{
    printf '//go:build !android && linux && amd64 && !musl\n\n'
    printf 'package sherpa_onnx\n\n'
    printf '// #cgo LDFLAGS: -Wl,--start-group'
    for a in "$FORK_DIR/$LIB_SUBDIR"/*.a; do
        printf ' ${SRCDIR}/%s/%s' "$LIB_SUBDIR" "$(basename "$a")"
    done
    printf ' -Wl,--end-group -l:libstdc++.a -static-libgcc -lm\n'
    printf 'import "C"\n'
} > "$FORK_DIR/build_linux_amd64.go"

# 5. Alternate module files so the default (dynamic) build stays untouched.
cp go.mod go.static.mod
cp go.sum go.static.sum
go mod edit -modfile=go.static.mod \
    -replace "github.com/k2-fsa/sherpa-onnx-go-linux=./build/sherpa-onnx-go-linux-static"

echo "Static build prepared. Build with:"
echo "  go build -modfile=go.static.mod -o transcribe ./cmd/transcribe"
