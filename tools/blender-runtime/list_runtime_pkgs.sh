#!/bin/sh
# Prints the Ubuntu packages providing every shared library Blender links against (run inside the build image).
set -e
ldd /opt/blender/blender | awk '/=> \//{print $3}' | while read -r lib; do
  real=$(readlink -f "$lib")
  for p in "$lib" "$real" "/usr$lib" "/usr$real"; do
    pkg=$(dpkg -S "$p" 2>/dev/null | head -n1 | cut -d: -f1) && [ -n "$pkg" ] && { echo "$pkg"; break; }
  done
done | sort -u | grep -v -E '^(libc6|libgcc-s1|libstdc\+\+6)$'
