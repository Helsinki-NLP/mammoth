check_under() {
  local base="$1"; shift
  local name f missing=0
  for name in "$@"; do
    f="$base/$name"
    [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; missing=1; }
  done
  (( missing == 0 )) || exit 1
}

log() {
    printf '%s %s\n' "[$BASENAME]" "$*";
}


