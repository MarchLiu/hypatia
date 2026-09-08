#!/usr/bin/env bash
# hypatia cross-build script
# Supports all major Rust targets via two backends:
#   1. cargo-zigbuild (default) — uses zig as cross C/C++ toolchain, no Docker needed
#   2. cross — uses Docker containers, best for CI
#
# Usage:
#   ./scripts/build.sh                          # build native target (release)
#   ./scripts/build.sh <target>                 # cross-build for one target
#   ./scripts/build.sh <target> <target> ...    # cross-build for multiple targets
#   ./scripts/build.sh all                      # build all supported targets
#   ./scripts/build.sh list                     # list supported targets
#   ./scripts/build.sh --backend cross <target> # use cross instead of zigbuild
#   ./scripts/build.sh --debug <target>         # debug build
#   ./scripts/build.sh --no-strip <target>      # skip strip step
#   ./scripts/build.sh --features postgres-backend # include PostgreSQL support
#
# Prerequisites:
#   - rustup + cargo
#   - For zigbuild:  cargo install cargo-zigbuild && python3 -m pip install ziglang
#   - For cross:     cargo install cross --git https://github.com/cross-rs/cross

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Supported targets ───

# Tier 1 + common Tier 2 targets where DuckDB/SQLite can realistically compile
SUPPORTED_TARGETS=(
    # Linux
    "x86_64-unknown-linux-gnu"
    "x86_64-unknown-linux-musl"
    "aarch64-unknown-linux-gnu"
    "aarch64-unknown-linux-musl"
    "armv7-unknown-linux-gnueabihf"
    "armv7-unknown-linux-musleabihf"
    "riscv64gc-unknown-linux-gnu"
    "s390x-unknown-linux-gnu"
    "powerpc64le-unknown-linux-gnu"

    # macOS
    "x86_64-apple-darwin"
    "aarch64-apple-darwin"

    # Windows
    "x86_64-pc-windows-msvc"
    "aarch64-pc-windows-msvc"
    "x86_64-pc-windows-gnu"

    # FreeBSD
    "x86_64-unknown-freebsd"
    "aarch64-unknown-freebsd"

    # NetBSD
    "x86_64-unknown-netbsd"

    # Android
    "aarch64-linux-android"
    "armv7-linux-androideabi"
    "x86_64-linux-android"

    # illumos
    "x86_64-unknown-illumos"
)

# ─── Defaults ───

BACKEND="zigbuild"     # zigbuild | cross
PROFILE="release"
STRIP="true"
NATIVE_TARGET="$(rustc -vV | sed -n 's/^host: //p')"

# ─── Arg parsing ───

TARGETS=()
FEATURE_FLAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --features)
            FEATURE_FLAGS=(--features "${2:?--features requires a value}")
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --debug)
            PROFILE="debug"
            shift
            ;;
        --no-strip)
            STRIP="false"
            shift
            ;;
        list)
            echo "Supported targets:"
            for t in "${SUPPORTED_TARGETS[@]}"; do
                if [[ "$t" == "$NATIVE_TARGET" ]]; then
                    echo "  $t  (native)"
                else
                    echo "  $t"
                fi
            done
            exit 0
            ;;
        all)
            TARGETS=("${SUPPORTED_TARGETS[@]}")
            shift
            ;;
        -h|--help)
            head -22 "$0" | tail -20
            exit 0
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Default: build native
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("$NATIVE_TARGET")
fi

# ─── Tool checks ───

check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "Error: $1 not found. $2"
        exit 1
    fi
}

install_target() {
    local target="$1"
    if ! rustup target list --installed | grep -q "$target"; then
        echo "Installing target: $target"
        rustup target add "$target"
    fi
}

# ─── Build functions ───

build_native() {
    local target="$1"
    echo "==> Native build for $target (profile: $PROFILE)"
    install_target "$target"

    local profile_flag=()
    if [[ "$PROFILE" == "release" ]]; then
        profile_flag=(--release)
    fi

    cargo build --target "$target" "${profile_flag[@]}" ${FEATURE_FLAGS[@]+"${FEATURE_FLAGS[@]}"} -p hypatia

    post_build "$target"
}

build_zigbuild() {
    local target="$1"
    echo "==> Zigbuild for $target (profile: $PROFILE)"
    check_tool "cargo-zigbuild" "Install with: cargo install cargo-zigbuild"

    install_target "$target"

    local profile_flag=()
    if [[ "$PROFILE" == "release" ]]; then
        profile_flag=(--release)
    fi

    cargo zigbuild --target "$target" "${profile_flag[@]}" ${FEATURE_FLAGS[@]+"${FEATURE_FLAGS[@]}"} -p hypatia

    post_build "$target"
}

build_cross() {
    local target="$1"
    echo "==> Cross build for $target (profile: $PROFILE)"
    check_tool "cross" "Install with: cargo install cross --git https://github.com/cross-rs/cross"

    install_target "$target"

    local profile_flag=()
    if [[ "$PROFILE" == "release" ]]; then
        profile_flag=(--release)
    fi

    cross build --target "$target" "${profile_flag[@]}" ${FEATURE_FLAGS[@]+"${FEATURE_FLAGS[@]}"} -p hypatia

    post_build "$target"
}

post_build() {
    local target="$1"
    local bin_name="hypatia"
    if [[ "$target" == *"-windows-"* ]]; then
        bin_name="hypatia.exe"
    fi

    local profile_dir="$PROFILE"
    local bin_path="target/$target/$profile_dir/$bin_name"

    if [[ -f "$bin_path" ]]; then
        local size
        size=$(du -h "$bin_path" | cut -f1)
        echo "    Output: $bin_path ($size)"

        if [[ "$STRIP" == "true" && "$PROFILE" == "release" ]]; then
            strip_binary "$target" "$bin_path"
        fi
    else
        echo "    Warning: binary not found at $bin_path"
    fi
    echo ""
}

strip_binary() {
    local target="$1"
    local bin="$2"

    # Don't strip macOS binaries (codesign issues) or Windows .exe
    if [[ "$target" == *"-apple-"* ]]; then
        return
    fi

    # Find appropriate strip tool
    local strip_tool="strip"
    if [[ "$target" == *"-linux-android"* ]]; then
        strip_tool="${ANDROID_NDK_HOME:-$ANDROID_HOME/ndk/$(ls "$ANDROID_HOME/ndk/" 2>/dev/null | tail -1)}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip"
        if [[ ! -f "$strip_tool" ]]; then
            strip_tool="llvm-strip"
        fi
    elif command -v "llvm-strip" &>/dev/null; then
        strip_tool="llvm-strip"
    fi

    if command -v "$strip_tool" &>/dev/null; then
        "$strip_tool" "$bin" 2>/dev/null && echo "    Stripped: $bin" || true
    fi
}

# ─── Main ───

echo "Hypatia cross-build"
echo "  Backend:  $BACKEND"
echo "  Profile:  $PROFILE"
echo "  Targets:  ${TARGETS[*]}"
echo ""

FAILED=()
SUCCEEDED=()

for target in "${TARGETS[@]}"; do
    # Validate target
    if [[ "$target" != "all" ]] && ! rustc --print target-list | grep -q "^${target}$"; then
        echo "Error: unknown target '$target'"
        echo "  Run './scripts/build.sh list' to see supported targets."
        FAILED+=("$target")
        continue
    fi

    if [[ "$target" == "$NATIVE_TARGET" ]]; then
        build_native "$target" && SUCCEEDED+=("$target") || FAILED+=("$target")
    elif [[ "$BACKEND" == "cross" ]]; then
        build_cross "$target" && SUCCEEDED+=("$target") || FAILED+=("$target")
    else
        build_zigbuild "$target" && SUCCEEDED+=("$target") || FAILED+=("$target")
    fi
done

# ─── Summary ───

echo "=========================================="
echo "  Build summary"
echo "=========================================="

if [[ ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo "  Succeeded: ${#SUCCEEDED[@]}"
    for t in "${SUCCEEDED[@]}"; do
        echo "    ✓ $t"
    done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed: ${#FAILED[@]}"
    for t in "${FAILED[@]}"; do
        echo "    ✗ $t"
    done
    exit 1
fi

echo ""
echo "All builds completed successfully."
