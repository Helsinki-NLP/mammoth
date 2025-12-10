#!/usr/bin/env bash
# datasets.sh — manage shared datasets under $SHARDATA
# Actions:
#   --check  (default)  Show space/quota + plan + cost estimates (no changes)
#   --get               Download/extract/clone the chosen dataset
#   --rm                Remove the dataset directory (prompts)
#   --stat              Print on-disk stats + costs for downloaded data
#
# Datasets (choose one):
#   europarl-3langs         → $SHARDATA/europarl/3langs
#   europarl-all|europarl   → $SHARDATA/europarl/all
#   vocab-opusTC.mul        → $SHARDATA/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
#   tatoeba                 → $SHARDATA/tatoeba/Tatoeba-Challenge (git)
#   tatoeba-subsets         → $SHARDATA/tatoeba/Tatoeba-Challenge/data/subsets (+ symlink: $SHARDATA/tatoeba/subsets)
#   uncorpus                → $SHARDATA/uncorpus (uses $SHARDATA/uncorpus/urls.txt manifest)
#   unpc-<src>-<tgt>[:fmt]  → $SHARDATA/uncorpus/<src>-<tgt>/<fmt> (fmt = moses|tmx) via OpusTools (no scraping)
#   opus100-<src>-<tgt>     → $SHARDATA/opus100/<src>-<tgt>  (generalized)
#   opus100-zeroshot        → $SHARDATA/opus100/zeroshot
#
# Examples:
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --check europarl-all
#   SYSTEM=puhti SHARDATA=/proj/xxx/shared ./datasets.sh --get   vocab-opusTC.mul
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --get   unpc-en-es
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --stat  tatoeba
#
# Notes:
# - Idempotent: re-running --get won’t overwrite existing extracted dirs; files guarded.
# - Prices: shows BU/TiB-month (for Puhti/Mahti defaults here) and LUMI TiB·h equivalents.
# - For LUMI capacities/budgets, set:  module load lumi-tools; export ACCOUNT=project_XXXXX
# - OpusTools path: we add ~/.local/bin to PATH; you can also set OPUS_READ_CMD to an explicit path.

set -euo pipefail

# Ensure user-installed scripts (pip --user) are found
export PATH="$HOME/.local/bin:$PATH"

# ─────────────── Storage cost config (edit/override if needed) ───────────────
: "${RATE_BU_PER_TIB_MONTH_LUMI:=}"       # LUMI uses TiB·hours natively → leave N/A unless you map to BU
: "${RATE_BU_PER_TIB_MONTH_PUHTI:=3600}"  # example default
: "${RATE_BU_PER_TIB_MONTH_MAHTI:=3600}"
: "${RATE_BU_PER_TIB_MONTH_ROIHU:=}"
: "${LUMI_BU_PER_TIBH:=}"                 # optional mapping TiB·h→BU on LUMI (rare)

# ───────────────────────────────── Helpers ───────────────────────────────────
: "${SHARDATA:?❌ SHARDATA is not set (root of shared datasets)}"
: "${SYSTEM:?❌ SYSTEM is not set (use: lumi|puhti|mahti|roihu)}"
SYSTEM="${SYSTEM,,}"

have(){ command -v "$1" >/dev/null 2>&1; }
log(){ printf '[datasets] %s\n' "$*"; }
warn(){ printf '[datasets][WARN] %s\n' "$*" >&2; }
err(){ printf '[datasets][ERR ] %s\n' "$*" >&2; }

bytes_to_h(){
  local b=${1:-0} k=1024 m=$((1024*1024)) g=$((1024*1024*1024)) t=$((1024*1024*1024*1024))
  if   (( b >= t )); then printf '%.2fT' "$(awk -v b="$b" 'BEGIN{print b/2^40}')"
  elif (( b >= g )); then printf '%.2fG' "$(awk -v b="$b" 'BEGIN{print b/2^30}')"
  elif (( b >= m )); then printf '%.2fM' "$(awk -v b="$b" 'BEGIN{print b/2^20}')"
  elif (( b >= k )); then printf '%.2fK' "$(awk -v b="$b" 'BEGIN{print b/2^10}')"
  else printf '%dB' "$b"; fi
}
bytes_to_gib(){ awk -v b="${1:-0}" 'BEGIN{printf "%.2f", b/2^30}'; }
bytes_to_tib(){ awk -v b="${1:-0}" 'BEGIN{printf "%.4f", b/2^40}'; }

content_length(){
  local url="$1" cl=""
  if have curl; then
    cl=$(curl -fsSI "$url" | awk -F': *' 'tolower($1)=="content-length"{print $2}' | tr -d '\r')
  elif have wget; then
    cl=$(wget -S --spider "$url" 2>&1 | awk -F': *' '/Content-Length/{print $2}' | tail -1)
  fi
  printf '%s' "${cl:-}"
}

show_space_quota(){
  echo
  echo "== Space & quota for $SYSTEM on $SHARDATA =="
  df -h "$SHARDATA" || true
  case "$SYSTEM" in
    lumi|puhti|mahti|roihu)
      if have lfs; then
        echo; echo "== lfs quota =="; lfs quota -h "$SHARDATA" || true
      fi
      ;;
  esac
  if have quota; then
    echo; echo "== user quota =="; quota -s || true
  fi
}

# BU per month for given bytes and rate (BU per TiB-month). Empty rate → "N/A".
bu_per_month(){
  local bytes="${1:-0}" rate="${2:-}"
  [[ -z "$rate" || "$rate" = 0 ]] && { printf "N/A"; return; }
  awk -v b="$bytes" -v r="$rate" 'BEGIN{printf "%.2f", (b/2^40)*r}'
}
# LUMI TiB·h for ~30 days (720h). If LUMI_BU_PER_TIBH is set, also show BU.
lumi_tibh_month_line(){
  local bytes="${1:-0}"
  awk -v b="$bytes" -v map="${LUMI_BU_PER_TIBH:-}" '
    BEGIN{
      tib  = b/2^40
      tbh  = tib*720.0
      if (map == "" || map == 0)
        printf "LUMI: TiB=%.4f  TiB·h/mo=%.2f  BU/mo=N/A\n", tib, tbh
      else
        printf "LUMI: TiB=%.4f  TiB·h/mo=%.2f  BU/mo=%.2f (%.3f BU/TiB·h)\n", tib, tbh, tbh*map, map
    }'
}
rate_for_system(){
  case "$SYSTEM" in
    lumi)  printf '%s' "${RATE_BU_PER_TIB_MONTH_LUMI}"  ;;
    puhti) printf '%s' "${RATE_BU_PER_TIB_MONTH_PUHTI}" ;;
    mahti) printf '%s' "${RATE_BU_PER_TIB_MONTH_MAHTI}" ;;
    roihu) printf '%s' "${RATE_BU_PER_TIB_MONTH_ROIHU}" ;;
    *)     printf '' ;;
  esac
}
print_storage_comparison(){
  local bytes="${1:-0}" gib tib rate bu
  gib="$(bytes_to_gib "$bytes")"
  tib="$(bytes_to_tib "$bytes")"
  rate="$(rate_for_system)"
  bu="$(bu_per_month "$bytes" "$rate")"
  echo
  echo "== storage comparison (≈30 days) =="
  printf "Size : %s GiB  (%.4f TiB)\n" "$gib" "$tib"
  if [[ -n "$rate" ]]; then
    printf "%s: BU/mo=%s  (rate %s BU/TiB-month)\n" "${SYSTEM^^}" "$bu" "$rate"
  else
    printf "%s: BU/mo=N/A (set RATE_BU_PER_TIB_MONTH_%s to see BU)\n" "${SYSTEM^^}" "${SYSTEM^^}"
  fi
  lumi_tibh_month_line "$bytes"
}

# ---------- TiB·hour helpers ----------
tibh_for_bytes_days() { awk -v b="${1:-0}" -v d="${2:-30}" 'BEGIN{ printf "%.2f", (b/2^40)*(d*24) }'; }
days_supported_by_tibh(){ awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN{t=b/2^40; if(t<=0){print "INF";exit} printf "%.0f", a/(t*24)}'; }

# ---------- LUMI storage helpers (quota & time budgets) ----------
size_to_bytes() {
  awk -v s="$1" 'BEGIN{
    unit=substr(s,length(s),1); num=s;
    if (unit ~ /[KkMmGgTt]/) num=substr(s,1,length(s)-1);
    if (unit ~ /[Tt]/) mul=2^40; else if (unit ~ /[Gg]/) mul=2^30;
    else if (unit ~ /[Mm]/) mul=2^20; else if (unit ~ /[Kk]/) mul=2^10; else mul=1;
    printf "%.0f", num*mul }'
}
get_lumi_quota_avail() {
  local acct="${ACCOUNT:-}"; [[ -z "$acct" ]] && return 1
  command -v lumi-quota >/dev/null 2>&1 || return 1
  lumi-quota -p "$acct" 2>/dev/null | awk '
    /\/scratch\/project_/  {split($2,a,"/"); print "scratch " a[2];}
    /\/projappl\/project_/ {split($2,a,"/"); print "projappl " a[2];}
    /\/flash\/project_/    {split($2,a,"/"); print "flash " a[2];}
  ' | while read -r area rest; do bytes=$(size_to_bytes "$rest"); printf "%s %s\n" "$area" "$bytes"; done
}
get_available_tibh() {
  if [[ -n "${AVAILABLE_TIBH:-}" ]]; then printf '%s' "$AVAILABLE_TIBH"; return; fi
  if [[ "${SYSTEM,,}" == "lumi" ]] && command -v lumi-allocations >/dev/null 2>&1; then
    local acct_opt=(); [[ -n "${ACCOUNT:-}" ]] && acct_opt=( -p "$ACCOUNT" )
    local rem_tibh
    rem_tibh="$(
      lumi-allocations "${acct_opt[@]}" 2>/dev/null \
      | awk -v acct="${ACCOUNT:-}" -F'|' '
          /^-+$/ || /Project[[:space:]]+\|/ { next }
          {
            if (acct != "") { left=$1; gsub(/^[ \t]+|[ \t]+$/,"",left); if (left !~ acct) next; }
            storage=$4; gsub(/^[ \t]+|[ \t]+$/,"",storage);
            split(storage,tok,/ +/); split(tok[1],ua,"/");
            used=ua[1]+0; alloc=ua[2]+0; rem_tb=alloc-used;
            printf "%.2f\n", rem_tb*0.90949470177; exit
          }')"
    [[ -n "$rem_tibh" ]] && { printf '%s' "$rem_tibh"; return; }
  fi
  printf ''
}

# ----- OpusTools (UNPC) helper -----
# Echoes the command to run (opus_read or python -m opustools_pkg.opus_read), returns 0 if available.
ensure_opus_tools() {
  if [ -n "${OPUS_READ_CMD:-}" ]; then
    command -v "$OPUS_READ_CMD" >/dev/null 2>&1 && { echo "$OPUS_READ_CMD"; return; }
  fi
  if command -v opus_read >/dev/null 2>&1; then
    echo "opus_read"; return
  fi
  if command -v python3 >/dev/null 2>&1 && python3 -c "import opustools_pkg" 2>/dev/null; then
    echo "python3 -m opustools_pkg.opus_read"; return
  fi
  if command -v python >/dev/null 2>&1 && python -c "import opustools_pkg" 2>/dev/null; then
    echo "python -m opustools_pkg.opus_read"; return
  fi
  return 1
}

# ────────────────────────── dataset definitions ──────────────────────────────
KIND=""; TARGET_DIR=""; declare -a URLS=(); EXTRACT_TGZ=0; EXTRACT_TAR=0; GIT_CLONE=""; URLS_MANIFEST=""
OPUS_TOOLS=0; OPUS_SRC=""; OPUS_TGT=""; OPUS_FMT=""

pick_dataset(){
  local name="$1"
  case "$name" in
    europarl-all|europarl)
      KIND="parallel"; TARGET_DIR="$SHARDATA/europarl/all"
      URLS=( "https://mammoth101.a3s.fi/europarl.tar.gz" ); EXTRACT_TGZ=1 ;;

    europarl-3langs)
      KIND="parallel"; TARGET_DIR="$SHARDATA/europarl/3langs"
      URLS=( "https://mammoth101.a3s.fi/europarl-3langs.tar.gz" ); EXTRACT_TGZ=1 ;;

    vocab-opusTC.mul)
      KIND="vocab"; TARGET_DIR="$SHARDATA/vocab"
      URLS=( "https://mammoth101.a3s.fi/opusTC.mul.64k.spm"
             "https://mammoth101.a3s.fi/opusTC.mul.vocab.onmt" )
      EXTRACT_TGZ=0 ;;

    tatoeba)
      KIND="parallel"; TARGET_DIR="$SHARDATA/tatoeba/Tatoeba-Challenge"
      # Example placeholder URL (devtest set); real repo is cloned by tatoeba-subsets
      URLS=("https://object.pouta.csc.fi/Tatoeba-Challenge-devtest/devtest.tar")
      EXTRACT_TAR=0 ;;

    tatoeba-subsets)
      KIND="parallel"; TARGET_DIR="$SHARDATA/tatoeba/Tatoeba-Challenge"
      GIT_CLONE="https://github.com/Helsinki-NLP/Tatoeba-Challenge.git"
      EXTRACT_TGZ=0 ;;

    uncorpus)
      KIND="parallel"; TARGET_DIR="$SHARDATA/uncorpus"
      URLS_MANIFEST="$TARGET_DIR/urls.txt"
      EXTRACT_TGZ=0; EXTRACT_TAR=0 ;;

    # UN Parallel Corpus via OpusTools (no scraping)
    # Usage: unpc-<src>-<tgt>[:moses|tmx]   (default fmt=moses)
    unpc-*-*|unpc-*-*:* )
      local spec="${name#unpc-}"
      local pair="${spec%%:*}"
      local fmt="${spec#*:}"
      [[ "$fmt" == "$spec" ]] && fmt="moses"
      local src="${pair%%-*}"
      local tgt="${pair#*-}"
      src="${src,,}"; tgt="${tgt,,}"; fmt="${fmt,,}"
      if [[ -z "$src" || -z "$tgt" || "$src" == "$tgt" ]]; then
        err "Bad UNPC pair: '$pair' (expected unpc-<src>-<tgt>[:moses|tmx])"; exit 2; fi
      if [[ ! "$fmt" =~ ^(moses|tmx)$ ]]; then
        err "Unknown format '$fmt' (use moses or tmx)"; exit 2; fi
      KIND="parallel"; OPUS_TOOLS=1; OPUS_SRC="$src"; OPUS_TGT="$tgt"; OPUS_FMT="$fmt"
      TARGET_DIR="$SHARDATA/uncorpus/${src}-${tgt}/${fmt}"
      EXTRACT_TGZ=0; EXTRACT_TAR=0 ;;

    # OPUS-100 generalization
    opus100-*-*)
      local pair="${name#opus100-}"; local src="${pair%%-*}"; local tgt="${pair#*-}"
      src="${src,,}"; tgt="${tgt,,}"
      if [[ -z "$src" || -z "$tgt" || "$src" == "$tgt" ]]; then err "Bad OPUS-100 pair: '$pair'"; exit 2; fi
      [[ "$src" =~ ^[a-z]{2,3}(-[a-z0-9]+)?$ && "$tgt" =~ ^[a-z]{2,3}(-[a-z0-9]+)?$ ]] || warn "Unusual codes: $src / $tgt"
      KIND="parallel"; TARGET_DIR="$SHARDATA/opus100/${src}-${tgt}"
      URLS=( "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-${src}-${tgt}-v1.0.tar.gz" )
      EXTRACT_TGZ=1 ;;

    opus100-zeroshot)
      KIND="parallel"; TARGET_DIR="$SHARDATA/opus100/zeroshot"
      URLS=( "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-zeroshot-v1.0.tar.gz" )
      EXTRACT_TGZ=1 ;;

    *)
      err "Unknown dataset: $name"; exit 2;;
  esac
}

# ─────────────────────────────── help & CLI ──────────────────────────────────
prog="${0##*/}"

list_datasets() {
  cat <<'DS'
Available datasets:
  europarl-3langs           → $SHARDATA/europarl/3langs
  europarl-all|europarl     → $SHARDATA/europarl/all
  vocab-opusTC.mul          → $SHARDATA/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
  tatoeba                   → $SHARDATA/tatoeba/Tatoeba-Challenge (git)
  tatoeba-subsets           → $SHARDATA/tatoeba/Tatoeba-Challenge/data/subsets
  uncorpus                  → $SHARDATA/uncorpus  (uses $SHARDATA/uncorpus/urls.txt manifest)
  unpc-<src>-<tgt>[:fmt]    → $SHARDATA/uncorpus/<src>-<tgt>/<fmt>  (fmt=moses|tmx, via OpusTools)
  opus100-<src>-<tgt>       → $SHARDATA/opus100/<src>-<tgt>
  opus100-zeroshot          → $SHARDATA/opus100/zeroshot
DS
}

usage() {
  cat <<EOF
Usage:
  $prog --check <dataset>   # show quota/space, plan, cost (default if <dataset> only)
  $prog --get   <dataset>   # download/extract/clone only that dataset
  $prog --rm    <dataset>   # remove dataset directory (asks to confirm)
  $prog --stat  <dataset>   # on-disk stats + costs
  $prog --list              # list supported datasets
  $prog --help              # this help

Required environment:
  SYSTEM=lumi|puhti|mahti|roihu
  SHARDATA=/path/to/shared/project/area
Optional:
  ACCOUNT=project_...       # for lumi-quota; on LUMI: module load lumi-tools
EOF
  list_datasets
}

# Parse args
ACTION=""; DATASET=""
case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  --list)       list_datasets; exit 0 ;;
  --check|--get|--rm|--stat)
    ACTION="${1#--}"; DATASET="${2:-}"; shift 2 || true ;;
  *)
    if [[ -n "${1:-}" && -z "${2:-}" && "${1:-}" != --* ]]; then
      ACTION="check"; DATASET="$1"; shift
    else
      echo "❌ Unknown command or wrong arguments." >&2
      usage; exit 2
    fi
    ;;
esac

[[ -n "$DATASET" ]] || { echo "❌ Please specify a dataset." >&2; usage; exit 2; }

pick_dataset "$DATASET"

# ───────────────────────────── file operations ───────────────────────────────
ensure_dir(){ [[ -d "$1" ]] || { mkdir -p "$1"; log "mkdir -p $1"; }; }

download_file(){
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then log "exists: $out"; return 0; fi
  if have curl; then
    log "curl -fL --retry 3 -o $out $url"
    curl -fL --retry 3 --retry-delay 2 -o "$out" "$url"
  elif have wget; then
    log "wget -O $out $url"
    wget -O "$out" "$url"
  else
    err "Need curl or wget to download"; exit 3
  fi
}

extract_tgz(){ local tgz="$1" to_dir="$2"; ensure_dir "$to_dir"; if compgen -G "$to_dir/*" >/dev/null; then log "looks extracted: $to_dir"; return 0; fi; log "extracting $tgz -> $to_dir"; tar -xzf "$tgz" -C "$to_dir"; }
extract_tar(){ local tarf="$1" to_dir="$2"; ensure_dir "$to_dir"; if compgen -G "$to_dir/*" >/dev/null; then log "looks extracted: $to_dir"; return 0; fi; log "extracting $tarf -> $to_dir"; tar -xf "$tarf" -C "$to_dir"; }

clone_or_update(){
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then
    log "git -C $dir fetch --all --prune"
    git -C "$dir" fetch --all --prune
  else
    ensure_dir "$(dirname "$dir")"
    log "git clone $url $dir"
    git clone "$url" "$dir"
  fi
}

# Manifest reader (for uncorpus)
read_manifest_urls(){
  local file="$1"; [[ -s "$file" ]] || return 1
  awk 'NF && $1 !~ /^#/' "$file"
}

dataset_stats(){
  local dir="$1"
  if [[ ! -d "$dir" ]]; then echo "No data at $dir"; return 0; fi
  echo "== stats for $dir =="
  du -sh "$dir" || true
  find "$dir" -type f | wc -l | awk '{print "files:",$1}'
}

estimate_download_size(){
  local total=0 sz
  local urls=("${URLS[@]:-}")
  if [[ -n "${URLS_MANIFEST:-}" && -s "${URLS_MANIFEST:-}" ]]; then
    while IFS= read -r u; do
      [[ -n "$u" ]] || continue
      urls+=("$u")
    done < <(awk 'NF && $1 !~ /^#/' "$URLS_MANIFEST")
  fi
  # avoid process substitution in strictly POSIX shells by guarding array length
  if ((${#urls[@]:-0} > 0)); then
    for u in "${urls[@]}"; do
      sz=$(content_length "$u" || true)
      [[ -n "$sz" ]] && total=$(( total + sz ))
    done
  fi
  printf '%s' "$total"
}

# ─────────────────────────── actions (idempotent) ────────────────────────────
if [[ "$ACTION" == "check" ]]; then
  show_space_quota
  echo
  echo "== plan =="
  echo "dataset: $DATASET  kind: ${KIND:-git}"
  echo "target : $TARGET_DIR"

  if [[ "${OPUS_TOOLS:-0}" = 1 ]]; then
    echo "source : OPUS/UNPC via OpusTools (opus_read)"
    echo "pair   : ${OPUS_SRC}-${OPUS_TGT}"
    echo "format : ${OPUS_FMT}  # output via -wm ${OPUS_FMT}; written with -w"
    echo "size   : N/A (generated/streamed by OpusTools; depends on pair/format)"
    print_storage_comparison 0
  elif [[ -n "${GIT_CLONE:-}" ]]; then
    echo "source : $GIT_CLONE (git)"
    echo "size   : N/A (git repo; depends on history)"
    [[ "$DATASET" == "tatoeba-subsets" ]] && echo "note   : subsets at: $TARGET_DIR/data/subsets (symlink: $SHARDATA/tatoeba/subsets)"
  else
    for u in "${URLS[@]:-}"; do echo "url    : $u"; done
    if [[ -n "${URLS_MANIFEST:-}" ]]; then
      echo "manifest: ${URLS_MANIFEST}  # one URL per line"
      [[ -s "$URLS_MANIFEST" ]] || echo "         (create this file with direct download URLs)"
    fi
    bytes=$(estimate_download_size || true)
    avail_bytes=$(df -PB1 "$SHARDATA" | awk 'NR==2{print $4}')
    if [[ -n "${bytes:-}" && "$bytes" -gt 0 && "$avail_bytes" -lt "$bytes" ]]; then
      echo "❌ Not enough free space: need ~$(bytes_to_h "$bytes"), have $(bytes_to_h "$avail_bytes")" >&2
      exit 1
    fi
    if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
      echo "size   : $(bytes_to_h "$bytes")  ($(bytes_to_gib "$bytes") GiB)"
      print_storage_comparison "$bytes"
    else
      echo "size   : N/A (no Content-Length available)"
      lumi_tibh_month_line 0
    fi

    # LUMI capacity advice from lumi-quota
    if [ "$(printf %s "${SYSTEM}" | tr '[:upper:]' '[:lower:]')" = "lumi" ] && \
       [ -n "${bytes:-}" ] && [ "$bytes" -gt 0 ]; then
      echo; echo "== capacity recommendation (from lumi-quota) =="
      if command -v lumi-quota >/dev/null 2>&1 && [ -n "${ACCOUNT:-}" ]; then
        best_area=""; best_fit=0; chosen_bytes=0
        tmpfile=$(mktemp) || exit 1
        get_lumi_quota_avail > "$tmpfile" || true
        while IFS=' ' read -r area avail; do
          [ -n "$area" ] || continue
          if [ "$avail" -gt "$bytes" ]; then
            if [ "$avail" -gt "$best_fit" ]; then best_fit="$avail"; best_area="$area"; fi
          fi
          [ "$avail" -gt "$chosen_bytes" ] && chosen_bytes="$avail"
          printf "  %-8s free: %8s (need ~%s)\n" "$area" "$(bytes_to_h "$avail")" "$(bytes_to_h "$bytes")"
        done < "$tmpfile"; rm -f "$tmpfile"
        if [ -n "$best_area" ]; then
          echo "✔ Recommend storing on: /$best_area (enough space)."
        else
          echo "❗ None of the areas have enough free capacity (largest ~$(bytes_to_h "$chosen_bytes"))."
        fi
      else
        echo "lumi-quota not available or ACCOUNT not set; module load lumi-tools; export ACCOUNT=<project>"
      fi
    fi

    # TiB·h budget
    if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
      avail_tibh="$(get_available_tibh)"; keep_tibh="$(tibh_for_bytes_days "$bytes" 30)"
      echo; echo "== storage budget (TiB·hours) =="
      if [[ -n "$avail_tibh" ]]; then
        echo "Project TiB·h (remaining): $avail_tibh"
        echo "This dataset for 30 days : $keep_tibh TiB·h"
        if awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{exit !(a<k)}'; then
          days_ok="$(days_supported_by_tibh "$avail_tibh" "$bytes")"
          echo "❗ Not enough for 30 days → keep ~${days_ok} days."
        else
          months_fit=$(awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{printf "%.1f", a/k}')
          echo "✔ Fits for ~${months_fit} month(s)."
        fi
      else
        echo "Project TiB·h: unknown (set AVAILABLE_TIBH=... or use lumi-allocations --storage)"
        echo "This dataset for 30 days: $keep_tibh TiB·h"
      fi
    fi
  fi
  echo
fi

case "$ACTION" in
  get)
    # OPUS/UNPC path (no scraping)
      if [[ "${OPUS_TOOLS:-0}" = 1 ]]; then
	  ensure_dir "$TARGET_DIR"
	  ensure_dir "$TARGET_DIR/.cache"
	  OPUS_READ_CMD_RESOLVED="$(ensure_opus_tools)" \
	      || { err "OpusTools not found. Try: module load python; pip install --user opustools-pkg"; exit 3; }

	  # Force remote OPUS object store; avoid site-local mirrors
	  OPUS_ENV=( env
		     OPUS_URL=https://object.pouta.csc.fi/OPUS
		     OPUS_CORPUS_URL=https://object.pouta.csc.fi/OPUS
		     OPUS_SOURCE_URL=https://object.pouta.csc.fi/OPUS
		     OPUS_ROOT="$TARGET_DIR/.cache"     # make OUR cache the 'root' so /proj/nlpl/... is never used
		   )

	  pair="${OPUS_SRC}-${OPUS_TGT}"
	  log "Fetching UNPC ${pair} (${OPUS_FMT}) → $TARGET_DIR"

	  # Common args: use our cache for both reading and downloading
	  OPUS_ARGS_COMMON=(
	      -d UNPC -r v1 -s "$OPUS_SRC" -t "$OPUS_TGT"
	      -dl "$TARGET_DIR/.cache"           # download dir
	      -rd "$TARGET_DIR/.cache"           # root dir to look into (prevents /proj/nlpl/…)
	      --suppress_prompts
	  )

	  opus_ok=0
	  if [ "$OPUS_FMT" = "moses" ]; then
	      "${OPUS_ENV[@]}" $OPUS_READ_CMD_RESOLVED \
			       "${OPUS_ARGS_COMMON[@]}" \
			       -wm moses \
			       -w  "$TARGET_DIR/${pair}.${OPUS_SRC}" "$TARGET_DIR/${pair}.${OPUS_TGT}" \
		  && opus_ok=1

	      # Post-condition: both files must exist and be non-empty
	      if (( opus_ok == 1 )) && [[ -s "$TARGET_DIR/${pair}.${OPUS_SRC}" && -s "$TARGET_DIR/${pair}.${OPUS_TGT}" ]]; then
		  log "done (--get $DATASET)"; exit 0
	      fi
	  else
	      "${OPUS_ENV[@]}" $OPUS_READ_CMD_RESOLVED \
			       "${OPUS_ARGS_COMMON[@]}" \
			       -wm tmx \
			       -w "$TARGET_DIR/${pair}.tmx" \
		  && opus_ok=1

	      if (( opus_ok == 1 )) && [[ -s "$TARGET_DIR/${pair}.tmx" ]]; then
		  log "done (--get $DATASET)"; exit 0
	      fi
	  fi

	  # -------- Fallback: fetch from object store directly (try several patterns) --------
	  warn "OpusTools did not produce files; trying direct URLs from OPUS object store…"

	  first_ok_url() {
	      # prints the first URL that returns HTTP 200 (empty if none)
	      for u in "$@"; do
		  code=$(curl -fsSIL -o /dev/null -w '%{http_code}' "$u" || true)
		  if [ "$code" = "200" ]; then printf '%s' "$u"; return 0; fi
	      done
	      return 1
	  }

	  if [ "$OPUS_FMT" = "tmx" ]; then
	      gz="$TARGET_DIR/${pair}.tmx.gz"
	      # Known layouts for UNPC on OPUS:
	      #   https://object.pouta.csc.fi/OPUS-UNPC/v1/tmx/en-es.tmx.gz
	      #   https://object.pouta.csc.fi/OPUS-UNPC/v1.0/tmx/en-es.tmx.gz
	      url="$(first_ok_url \
      "https://object.pouta.csc.fi/OPUS-UNPC/v1/tmx/${pair}.tmx.gz" \
      "https://object.pouta.csc.fi/OPUS-UNPC/v1.0/tmx/${pair}.tmx.gz" \
      "https://object.pouta.csc.fi/OPUS/UNPC/v1/tmx/${pair}.tmx.gz" \
      "https://object.pouta.csc.fi/OPUS/UNPC/v1.0/tmx/${pair}.tmx.gz")" || url=""
	      if [ -z "$url" ]; then
		  err "No TMX URL resolved for ${pair} (tried OPUS-UNPC v1/v1.0)."; exit 2
	      fi
	      download_file "$url" "$gz" || { err "Direct download failed: $url"; exit 2; }
	      gunzip -f "$gz" || { err "gunzip failed: $gz"; exit 2; }
	      [ -s "$TARGET_DIR/${pair}.tmx" ] || { err "Missing: $TARGET_DIR/${pair}.tmx"; exit 2; }
	      log "done (--get $DATASET)"; exit 0

	  else
	      zip="$TARGET_DIR/${pair}.txt.zip"
	      # Known layouts for Moses:
	      #   https://object.pouta.csc.fi/OPUS-UNPC/v1/moses/en-es.txt.zip
	      #   https://object.pouta.csc.fi/OPUS-UNPC/v1.0/moses/en-es.txt.zip
	      url="$(first_ok_url \
      "https://object.pouta.csc.fi/OPUS-UNPC/v1/moses/${pair}.txt.zip" \
      "https://object.pouta.csc.fi/OPUS-UNPC/v1.0/moses/${pair}.txt.zip" \
      "https://object.pouta.csc.fi/OPUS/UNPC/v1/moses/${pair}.txt.zip" \
      "https://object.pouta.csc.fi/OPUS/UNPC/v1.0/moses/${pair}.txt.zip")" || url=""
	      if [ -z "$url" ]; then
		  err "No Moses URL resolved for ${pair} (tried OPUS-UNPC v1/v1.0)."; exit 2
	      fi
	      download_file "$url" "$zip" || { err "Direct download failed: $url"; exit 2; }

	      tmpu="$TARGET_DIR/.cache/unzip.$$"
	      mkdir -p "$tmpu"
	      unzip -o -q "$zip" -d "$tmpu" || { err "unzip failed: $zip"; rm -rf "$tmpu"; exit 2; }

	      # find aligned files (*.SRC and *.TGT) and move/rename to ${pair}.* in TARGET_DIR
	      srccand="$(find "$tmpu" -type f -iname "*.${OPUS_SRC}" | head -n1 || true)"
	      tgtcand="$(find "$tmpu" -type f -iname "*.${OPUS_TGT}" | head -n1 || true)"
	      if [ -n "$srccand" ] && [ -n "$tgtcand" ]; then
		  mv -f "$srccand" "$TARGET_DIR/${pair}.${OPUS_SRC}"
		  mv -f "$tgtcand" "$TARGET_DIR/${pair}.${OPUS_TGT}"
		  rm -rf "$tmpu"
		  if [ ! -s "$TARGET_DIR/${pair}.${OPUS_SRC}" ] || [ ! -s "$TARGET_DIR/${pair}.${OPUS_TGT}" ]; then
		      err "Downloaded ZIP didn’t contain non-empty ${OPUS_SRC}/${OPUS_TGT} files"; exit 2
		  fi
		  log "done (--get $DATASET)"; exit 0
	      else
		  rm -rf "$tmpu"
		  err "Could not identify .${OPUS_SRC}/.${OPUS_TGT} files in ZIP (inspect with: unzip -l '$zip')"
		  exit 2
	      fi
	  fi

      fi

    # Git clone path
    if [[ -n "${GIT_CLONE:-}" ]]; then
      clone_or_update "$GIT_CLONE" "$TARGET_DIR"
      if [[ "$DATASET" == "tatoeba-subsets" ]]; then
        ensure_dir "$SHARDATA/tatoeba"
        ln -snf "$TARGET_DIR/data/subsets" "$SHARDATA/tatoeba/subsets"
        log "linked: $SHARDATA/tatoeba/subsets → $TARGET_DIR/data/subsets"
      fi
      log "done (--get $DATASET)"
      exit 0
    fi

    # URL download/extract path
    ensure_dir "$TARGET_DIR"
    tmp="$SHARDATA/.tmp-datasets"; ensure_dir "$tmp"
    declare -a urls=("${URLS[@]:-}")
    if [[ -n "${URLS_MANIFEST:-}" && -s "${URLS_MANIFEST:-}" ]]; then
      while IFS= read -r u; do
        [[ -n "$u" ]] || continue
        urls+=("$u")
      done < <(awk 'NF && $1 !~ /^#/' "$URLS_MANIFEST")
    fi

    if ((${#urls[@]:-0} == 0)); then
      warn "No URLs to download (empty list)."
    fi

    for u in "${urls[@]:-}"; do
      [[ -n "$u" ]] || continue
      base="$(basename "$u")"; part="$tmp/$base"
      download_file "$u" "$part"
      if (( EXTRACT_TGZ==1 )); then
        extract_tgz "$part" "$TARGET_DIR"
      elif (( EXTRACT_TAR==1 )); then
        extract_tar "$part" "$TARGET_DIR"
      else
        dest="$TARGET_DIR/$base"
        [[ -f "$dest" ]] || { mv "$part" "$dest"; log "moved $base -> $dest"; }
      fi
    done

    if [[ "$DATASET" = "vocab-opusTC.mul" ]]; then
      [[ -f "$TARGET_DIR/opusTC.mul.64k.spm" ]] || warn "missing .spm in $TARGET_DIR"
      [[ -f "$TARGET_DIR/opusTC.mul.vocab.onmt" ]] || warn "missing .onmt in $TARGET_DIR"
    fi
    log "done (--get $DATASET)"
    ;;

  rm)
    if [[ -e "$TARGET_DIR" ]]; then
      read -r -p "Remove $TARGET_DIR ? [y/N] " ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then rm -rf "$TARGET_DIR"; log "removed $TARGET_DIR"; else log "aborted"; fi
    else
      log "nothing to remove: $TARGET_DIR"
    fi
    ;;

  stat)
    dataset_stats "$TARGET_DIR"
    if du -sb "$TARGET_DIR" >/dev/null 2>&1; then
      bytes_on_disk=$(du -sb "$TARGET_DIR" | awk '{print $1}')
      print_storage_comparison "$bytes_on_disk"

      # LUMI capacity advice for existing data
      if [ "$(printf %s "${SYSTEM}" | tr '[:upper:]' '[:lower:]')" = "lumi" ] && \
         [ -n "${bytes_on_disk:-}" ] && [ "$bytes_on_disk" -gt 0 ]; then
        echo; echo "== capacity recommendation (from lumi-quota) =="
        if command -v lumi-quota >/dev/null 2>&1 && [ -n "${ACCOUNT:-}" ]; then
          best_area=""; best_fit=0; chosen_bytes=0
          tmpfile=$(mktemp) || exit 1
          get_lumi_quota_avail > "$tmpfile" || true
          while IFS=' ' read -r area avail; do
            [ -n "$area" ] || continue
            if [ "$avail" -gt "$bytes_on_disk" ]; then
              if [ "$avail" -gt "$best_fit" ]; then best_fit="$avail"; best_area="$area"; fi
            fi
            [ "$avail" -gt "$chosen_bytes" ] && chosen_bytes="$avail"
            printf "  %-8s free: %8s (need ~%s)\n" "$area" "$(bytes_to_h "$avail")" "$(bytes_to_h "$bytes_on_disk")"
          done < "$tmpfile"; rm -f "$tmpfile"
          if [ -n "$best_area" ]; then
            echo "✔ Recommend storing on: /$best_area."
          else
            echo "❗ None of the areas have enough free capacity (largest ~$(bytes_to_h "$chosen_bytes"))."
          fi
        else
          echo "lumi-quota not available or ACCOUNT not set; module load lumi-tools; export ACCOUNT=<project>"
        fi
      fi

      if [[ -n "${bytes_on_disk:-}" && "$bytes_on_disk" -gt 0 ]]; then
        avail_tibh="$(get_available_tibh)"; keep_tibh="$(tibh_for_bytes_days "$bytes_on_disk" 30)"
        echo; echo "== storage budget (TiB·hours) =="
        if [[ -n "$avail_tibh" ]]; then
          echo "Project TiB·h (remaining): $avail_tibh"
          echo "Current data for 30 days : $keep_tibh TiB·h"
          if awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{exit !(a<k)}'; then
            days_ok="$(days_supported_by_tibh "$avail_tibh" "$bytes_on_disk")"
            echo "❗ Not enough for 30 days → keep ~${days_ok} days."
          else
            months_fit=$(awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{printf "%.1f", a/k}')
            echo "✔ Fits for ~${months_fit} month(s)."
          fi
        else
          echo "Project TiB·h: unknown (set AVAILABLE_TIBH=... or use lumi-allocations --storage)"
          echo "Current data for 30 days: $keep_tibh TiB·h"
        fi
      fi
    else
      echo "== storage comparison (≈30 days) =="
      echo "Size : N/A (du -sb not available on this system)"
      echo; lumi_tibh_month_line 0
    fi
    ;;
esac

exit 0
# =============================================================================
# datasets.sh — Tips & Notes
# -----------------------------------------------------------------------------
# • Idempotent behavior
#   - Re-running --get won’t overwrite existing files:
#     * archives download once to $SHARDATA/.tmp-datasets
#     * extraction runs only if the target directory looks empty
#     * git datasets run `git fetch` if already cloned
#
# • Space & quota visibility
#   - Always shows filesystem space:   df -h "$SHARDATA"
#   - On CSC Lustre (LUMI/Puhti/Mahti/Roihu): tries `lfs quota -h "$SHARDATA"`
#   - Also shows user quota if `quota -s` exists
#
# • Size estimation before download
#   - Uses HTTP Content-Length via curl/wget HEAD; if missing, prints “N/A”
#   - Actual on-disk size for --stat uses `du -sb` (bytes) when available
#
# • Storage cost comparison (≈30 days)
#   - Prints BOTH:
#       1) BU/month using per-system rate “BU per TiB-month”
#       2) LUMI-native TiB·hours/month (= TiB * 720 h) and optional BU if you
#          set a conversion `LUMI_BU_PER_TIBH`
#   - Configure rates at the top (or via env):
#       RATE_BU_PER_TIB_MONTH_{LUMI,PUHTI,MAHTI,ROIHU}
#       # Optional, if your project has a mapping:
#       LUMI_BU_PER_TIBH   # BU per TiB·hour (EXAMPLE ONLY; may be unknown)
#   - If a rate/mapping is unset, script prints “N/A” to avoid false precision
#   - (Optional) If your site has free tiers (e.g., first 1 TiB free), you can
#     subtract that before computing BU; not enabled by default
#
# • Targets / where files go
#   - europarl-3langs   → $SHARDATA/europarl/3langs
#   - vocab-opusTC.mul  → $SHARDATA/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
#   - tatoeba           → $SHARDATA/tatoeba/Tatoeba-Challenge (git)
#   - opus100-de-en     → $SHARDATA/opus100/de-en
#   - opus100-zeroshot  → $SHARDATA/opus100/zeroshot
#
# • Actions / flags
#   - --check (default): print quota/space, download plan, and cost estimates
#   - --get            : download/extract/clone the chosen dataset
#   - --rm             : remove the dataset directory (asks to confirm)
#   - --stat           : show on-disk stats + cost comparison
#
# • Usage examples
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --check europarl-3langs
#   SYSTEM=puhti SHARDATA=/proj/xxx/shared ./datasets.sh --g
#
#   Just inspect plan + cost first
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --check europarl-all
#
#   Download/extract full Europarl
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --get   europarl-all
#
#   Stats + cost after it’s on disk
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --stat  europarl
#




