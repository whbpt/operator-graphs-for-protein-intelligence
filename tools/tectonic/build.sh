#!/bin/sh
set -eu

tool_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

case "${1-}" in
  -V|--version)
    if [ -x "$tool_dir/tectonic-bin" ]; then
      exec "$tool_dir/tectonic-bin" --version
    fi
    exec tectonic --version
    ;;
esac

if [ -x "$tool_dir/tectonic-bin" ] && [ -f "$tool_dir/default_bundle.zip" ]; then
  export TECTONIC_CACHE_DIR="$tool_dir/cache"
  exec "$tool_dir/tectonic-bin" -X compile -C \
    -b "$tool_dir/default_bundle.zip" "$@"
fi

if command -v tectonic >/dev/null 2>&1; then
  exec tectonic -X compile "$@"
fi

echo "Tectonic is required: https://tectonic-typesetting.github.io/" >&2
exit 127
