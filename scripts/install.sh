#!/bin/sh
# Installs the prebuilt hypatia binary from GitHub Releases:
#
#   curl -fsSL https://raw.githubusercontent.com/MarchLiu/hypatia/main/scripts/install.sh | sh
#
# Environment:
#   HYPATIA_VERSION       release tag to install, e.g. v4.0.0 (default: the latest release)
#   HYPATIA_INSTALL_DIR   where to put the binary (default: ~/.local/bin)
#   HYPATIA_REPO          GitHub repository to download from (default: MarchLiu/hypatia)
#   HYPATIA_DOWNLOAD_URL  http(s) URL of a directory holding the release files, used instead
#                         of GitHub; HYPATIA_VERSION and HYPATIA_REPO are then ignored
#
# A file downloaded with curl carries no quarantine attribute, so macOS runs the unsigned
# binary without a Gatekeeper prompt.
set -eu

REPO="${HYPATIA_REPO:-MarchLiu/hypatia}"
# Where no prebuilt binary works, building hypatia alone does not help: cargo fetches the same
# prebuilt ONNX Runtime, or has none for the target.
FROM_SOURCE="build ONNX Runtime yourself and hypatia against it (ORT_LIB_PATH; see https://ort.pyke.io/setup/linking)"

say() { printf '%s\n' "$*"; }
fail() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}
need() { command -v "$1" >/dev/null 2>&1 || fail "$1 is required"; }

# Sets `target` to the release build for this machine, or explains why there is none.
detect_target() {
    os=$(uname -s)
    arch=$(uname -m)
    if [ "$os" = Linux ] && ldd --version 2>&1 | grep -qi musl; then
        fail "the prebuilt Linux binaries need glibc, and this system uses musl; $FROM_SOURCE"
    fi
    case "$os/$arch" in
    Darwin/arm64)
        target=aarch64-apple-darwin
        ;;
    Darwin/x86_64)
        # A shell under Rosetta on an Apple Silicon Mac still gets the native build.
        if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" = 1 ]; then
            target=aarch64-apple-darwin
        else
            fail "there is no prebuilt ONNX Runtime for Intel Macs, so no prebuilt hypatia; $FROM_SOURCE"
        fi
        ;;
    Linux/x86_64 | Linux/amd64)
        # ort's prebuilt ONNX Runtime needs AVX2. Without it semantic search dies of an illegal
        # instruction, long after `hypatia --version` worked.
        if [ -r /proc/cpuinfo ] && ! grep -qw avx2 /proc/cpuinfo; then
            fail "this CPU lacks AVX2, which the prebuilt ONNX Runtime needs; $FROM_SOURCE"
        fi
        target=x86_64-unknown-linux-gnu
        ;;
    Linux/aarch64 | Linux/arm64)
        target=aarch64-unknown-linux-gnu
        ;;
    MINGW* | MSYS* | CYGWIN*)
        fail "on Windows, download hypatia-x86_64-pc-windows-msvc.tar.gz from https://github.com/$REPO/releases"
        ;;
    *)
        fail "there is no prebuilt hypatia for $os $arch, nor a prebuilt ONNX Runtime; $FROM_SOURCE"
        ;;
    esac
}

# Downloads $1 to $2. Returns 1 when the server has no such file (HTTP 404); fails on anything
# else, so a network problem is not mistaken for a missing file.
fetch() {
    code=$(curl -sSL --retry 3 --proto "$proto" --proto-redir "$proto" --tlsv1.2 \
        -o "$2" -w '%{http_code}' "$1" 2>"$tmp/curl.err") ||
        fail "could not download $1: $(cat "$tmp/curl.err")"
    case "$code" in
    200) return 0 ;;
    404)
        rm -f "$2"
        return 1
        ;;
    *) fail "could not download $1: the server answered HTTP $code" ;;
    esac
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        return 1
    fi
}

main() {
    need curl
    need mktemp
    detect_target
    dir="${HYPATIA_INSTALL_DIR:-$HOME/.local/bin}"
    proto='=https'
    if [ -n "${HYPATIA_DOWNLOAD_URL:-}" ]; then
        base="${HYPATIA_DOWNLOAD_URL%/}"
        proto='=http,https'
    elif [ -n "${HYPATIA_VERSION:-}" ]; then
        base="https://github.com/$REPO/releases/download/$HYPATIA_VERSION"
    else
        base="https://github.com/$REPO/releases/latest/download"
    fi

    mkdir -p "$dir" 2>/dev/null && [ -w "$dir" ] ||
        fail "cannot write to $dir; set HYPATIA_INSTALL_DIR to a directory you can write to"
    tmp=$(mktemp -d)
    staged=
    trap 'rm -rf "$tmp"; if [ -n "$staged" ]; then rm -f "$staged"; fi' EXIT
    trap 'exit 1' INT TERM

    # Releases ship hypatia-<target>.tar.gz with a .sha256; the first ones shipped a gzipped
    # binary without one.
    asset=
    for name in "hypatia-$target.tar.gz" "hypatia-$target.gz"; do
        if fetch "$base/$name" "$tmp/$name"; then
            asset=$name
            break
        fi
    done
    if [ -z "$asset" ]; then
        if [ -n "${HYPATIA_DOWNLOAD_URL:-}${HYPATIA_VERSION:-}" ]; then
            fail "$base has no hypatia-$target.tar.gz"
        fi
        fail "the latest release has no hypatia-$target.tar.gz; set HYPATIA_VERSION to a release tag that has one (https://github.com/$REPO/releases)"
    fi
    say "Downloaded $base/$asset"

    if fetch "$base/$asset.sha256" "$tmp/$asset.sha256"; then
        expected=$(cut -d' ' -f1 <"$tmp/$asset.sha256")
        actual=$(sha256_of "$tmp/$asset") || fail "sha256sum or shasum is needed to verify $asset"
        [ "$expected" = "$actual" ] || fail "$asset does not match its published sha256; nothing was installed"
    elif [ "${asset%.tar.gz}" != "$asset" ]; then
        fail "$asset has no published sha256, so it cannot be verified; nothing was installed"
    else
        say "note: $asset is from an early release without a sha256; installing it unverified"
    fi

    case "$asset" in
    *.tar.gz)
        need tar
        tar -xzf "$tmp/$asset" -C "$tmp" hypatia || fail "could not unpack hypatia from $asset"
        ;;
    *)
        need gzip
        gzip -dc "$tmp/$asset" >"$tmp/hypatia" || fail "could not unpack $asset"
        ;;
    esac

    # Stage inside the destination directory: the binary is tried where it will live (a noexec
    # /tmp would refuse it), and then renamed into place in one step.
    staged=$(mktemp "$dir/.hypatia.XXXXXX")
    cp "$tmp/hypatia" "$staged"
    chmod 755 "$staged"
    # Refuse a binary that cannot start here (an older glibc, no libstdc++ or OpenSSL 3)
    # before it replaces anything.
    if ! version=$("$staged" --version 2>&1); then
        fail "the downloaded binary does not run on this system: $version"
    fi
    mv -f "$staged" "$dir/hypatia"
    staged=
    say "Installed $version to $dir/hypatia"
    case ":$PATH:" in
    *":$dir:"*) ;;
    *) say "$dir is not on your PATH; add it, for example: export PATH=\"$dir:\$PATH\"" ;;
    esac
    say "Next: hypatia init"
}

main "$@"
