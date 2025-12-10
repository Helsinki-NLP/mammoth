#!/bin/sh
# showjobs.sh — POSIX shell, width-aware listing of job dirs

set -eu

: "${PROJHOME:?❌ PROJHOME not set}"

data_dir="$PROJHOME/data"
if [ ! -d "$data_dir" ]; then
  echo "❌ Not a directory: $data_dir" >&2
  exit 1
fi

# Build a tab-separated list: "<name>\t<fullpath>"
# Use globbing instead of bashisms; quotes keep spaces safe.
list_dirs() {
  found=0
  for d in "$data_dir"/*/ ; do
    [ -d "$d" ] || continue
    found=1
    d=${d%/}
    name=$(basename "$d")
    printf '%s\t%s\n' "$name" "$d"
  done
  # If nothing matched, the glob stays literal; suppress that.
  [ $found -eq 1 ] || true
}

# Feed to awk to compute widths and print a pretty table
list_dirs |
    awk -v P="$PROJHOME" -F '\t' -v hdr1="JOB_NAME" -v hdr2="JOB_DIR" -v datadir="$data_dir" '
BEGIN { col1=length(hdr1); col2=length(hdr2); n=0 }
NF>=2 {
  names[n]=$1; paths[n]=$2;
  sub("^" P, "$PROJHOME", paths[n]); 
  if (length($1) > col1) col1=length($1);
  if (length(paths[n]) > col2) col2=length(paths[n]);
  n++
}
function dash(len,    i,s){ s=""; for(i=0;i<len;i++) s=s "-"; return s }

END {
  print ""
  printf "%-*s | %s\n", col1, hdr1, hdr2
  printf "%s-+-%s\n", dash(col1), dash(col2)
  if (n==0) {
    printf "%-*s | %s\n", col1, "(no job dirs)", datadir
  } else {
    for (i=0; i<n; i++) {
      printf "%-*s | %s\n", col1, names[i], paths[i]
    }
  }
  printf "%s-+-%s\n", dash(col1), dash(col2)
  print ""
}'

