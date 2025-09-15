#!/usr/bin/env bash
# datasets.sh — manage shared datasets under $SHARDATA
# Actions:
#   --check  (default)  Show space/quota + plan + cost estimates (no changes)
#   --get               Download/extract/clone the chosen dataset
#   --rm                Remove the dataset directory (prompts)
#   --stat              Print on-disk stats + costs for already-downloaded data
#
# Datasets (choose one):
#   europarl-3langs   → $SHARDATA/europarl/3langs
#   vocab-opusTC.mul  → $SHARDATA/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
#   tatoeba           → $SHARDATA/tatoeba/Tatoeba-Challenge (git)
#   opus100-de-en     → $SHARDATA/opus100/de-en
#   opus100-zeroshot  → $SHARDATA/opus100/zeroshot
#
# Examples:
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --check europarl-3langs
#   SYSTEM=puhti SHARDATA=/proj/yyy/shared ./datasets.sh --get   vocab-opusTC.mul
#   SYSTEM=lumi  SHARDATA=/proj/xxx/shared ./datasets.sh --stat  tatoeba
#   SYSTEM=roihu SHARDATA=/proj/zzz/shared ./datasets.sh --rm    opus100-de-en
#
# Notes:
# - Idempotent: re-running --get won’t overwrite existing files; extracts only if needed.
# - Quota/space: tries `lfs quota`, falls back to `quota` and always shows `df -h`.
# - Costs: shows BOTH BU/TiB-month (per-system rate below) and LUMI native TiB·hours.
#   Override the rate variables here (or via environment) if your site updates them.

set -euo pipefail

# ─────────────── Storage cost config (edit/override if needed) ───────────────
# Billing Units (BU) per TiB-month (≈ 30 days) for *project* storage:
# These are best-known defaults; override via environment if your project uses
# different rates or if the site has updated pricing.
: "${RATE_BU_PER_TIB_MONTH_LUMI:=}"    # LUMI uses TiB·hours, leave empty unless you have a BU mapping
: "${RATE_BU_PER_TIB_MONTH_PUHTI:=3600}"  # ≈ 5 BU/TiB-hour → 5 * 720 = 3600 BU/TiB-month (example/default)
: "${RATE_BU_PER_TIB_MONTH_MAHTI:=3600}"  # same as Puhti by default
: "${RATE_BU_PER_TIB_MONTH_ROIHU:=}"      # fill in if applicable at your site

# Optional mapping from LUMI TiB·hour → BU (project-specific, if any)
: "${LUMI_BU_PER_TIBH:=}"  # e.g. export LUMI_BU_PER_TIBH=1.2  (EXAMPLE ONLY; leave empty if unknown)

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
		echo
		echo "== lfs quota =="
		lfs quota -h "$SHARDATA" || true
	    fi
	    ;;
    esac
    if have quota; then
	echo
	echo "== user quota =="
	quota -s || true
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

# ---------- TiB·hour (LUMI-style) budget helpers ----------
# Optional env override: AVAILABLE_TIBH sets your project's *remaining* TiB·hours
#   export AVAILABLE_TIBH=1234.5
#
# On LUMI you may also try to autodetect via site tools if available.
# If detection fails, we fall back to AVAILABLE_TIBH (or "unknown").

# Compute TiB·h over D days for size in bytes
tibh_for_bytes_days() {
  local bytes="${1:-0}" days="${2:-30}"
  awk -v b="$bytes" -v d="$days" 'BEGIN{ printf "%.2f", (b/2^40)*(d*24) }'
}

# Given AVAIL TiB·h and BYTES, how many days can we keep it?
days_supported_by_tibh() {
  local avail="${1:-0}" bytes="${2:-0}"
  awk -v a="$avail" -v b="$bytes" 'BEGIN{
    tib = b/2^40; if (tib<=0){print "INF"; exit}
    printf "%.0f", a/(tib*24)
  }'
}


# ---------- LUMI storage helpers (quota & time budgets) ----------

# Convert sizes like "479T", "1.1G", "8.2K" to bytes
size_to_bytes() {
  # accepts plain integer bytes too
  awk -v s="$1" '
    function pow(a,b){return a^b}
    BEGIN{
      unit=substr(s, length(s), 1)
      num = s
      if (unit ~ /[KkMmGgTt]/) { num = substr(s, 1, length(s)-1) }
      if (unit ~ /[Tt]/) mul=2^40; else if (unit ~ /[Gg]/) mul=2^30;
      else if (unit ~ /[Mm]/) mul=2^20; else if (unit ~ /[Kk]/) mul=2^10;
      else mul=1;
      printf "%.0f", num*mul
    }'
}

# Parse: lumi-quota -p "$ACCOUNT"
# Returns 3 lines (if present): "scratch <avail_bytes>", "projappl <avail_bytes>", "flash <avail_bytes>"
get_lumi_quota_avail() {
  local acct="${ACCOUNT:-}"
  [[ -z "$acct" ]] && return 1
  command -v lumi-quota >/dev/null 2>&1 || return 1

  # expected table like:
  # /scratch/project_XXXX   479T/550T ...
  lumi-quota -p "$acct" 2>/dev/null | awk '
    /\/scratch\/project_/  {split($2,a,"/"); print "scratch " a[2];}
    /\/projappl\/project_/ {split($2,a,"/"); print "projappl " a[2];}
    /\/flash\/project_/    {split($2,a,"/"); print "flash " a[2];}
  ' | while read -r area rest; do
        bytes=$(size_to_bytes "$rest")
        printf "%s %s\n" "$area" "$bytes"
      done
}

# Remaining TiB·hours for the project (LUMI)
# Order of precedence:
#   1) AVAILABLE_TIBH env (if set)
#   2) lumi-allocations [-p $ACCOUNT] parsed from the "Storage (used/allocated)" column
#      -> remaining = allocated - used (reported in TB·hours) → convert to TiB·hours
#   3) empty string if unavailable
get_available_tibh() {
  # 1) explicit env wins
  if [[ -n "${AVAILABLE_TIBH:-}" ]]; then
    printf '%s' "$AVAILABLE_TIBH"
    return
  fi

  # 2) LUMI helper (text table)
  if [[ "${SYSTEM,,}" == "lumi" ]] && command -v lumi-allocations >/dev/null 2>&1; then
    local acct_opt=()
    [[ -n "${ACCOUNT:-}" ]] && acct_opt=( -p "$ACCOUNT" )

    # Grab the line with the project row; if ACCOUNT is given, match that row; else take the first data row.
    # Then cut Storage column (field 4), extract "used/allocated", compute remaining TB·h, convert to TiB·h.
    local rem_tibh
    rem_tibh="$(
      lumi-allocations "${acct_opt[@]}" 2>/dev/null \
      | awk -v acct="${ACCOUNT:-}" -F'|' '
          BEGIN{ OFS=" "; }
          # skip header/underline rows
          /^-+$/ || /Project[[:space:]]+\|/ { next }
          {
            # If ACCOUNT is set, only accept the row that begins with it (left column).
            if (acct != "") {
              # Trim left column (project id)
              left=$1; gsub(/^[ \t]+|[ \t]+$/,"",left);
              if (left !~ acct) next;
            } else {
              # If no account given, just take the first data row
            }

            # Storage column is field 4: e.g. "   645756/4000000  (16.1%) TB/hours"
            storage=$4; gsub(/^[ \t]+|[ \t]+$/,"",storage);
            # The first token should be "used/allocated"
            split(storage, tok, /[ \t]+/);
            split(tok[1], ua, "/");
            used = ua[1] + 0;
            alloc = ua[2] + 0;
            rem_tb_hours = alloc - used;   # remaining in TB·hours
            # Convert TB·hours → TiB·hours. 1 TB = 10^12 bytes; 1 TiB = 2^40 bytes.
            # factor = 10^12 / 2^40 ≈ 0.90949470177
            rem_tib_hours = rem_tb_hours * 0.90949470177;
            printf "%.2f\n", rem_tib_hours;
            exit
          }
        '
    )"
    if [[ -n "$rem_tibh" ]]; then
      printf '%s' "$rem_tibh"
      return
    fi
  fi

  # 3) unknown
  printf ''
}


# ────────────────────────── dataset definitions ──────────────────────────────
KIND=""; TARGET_DIR=""; declare -a URLS=(); EXTRACT_TGZ=0; EXTRACT_TAR=0; GIT_CLONE=""

pick_dataset(){
  local name="$1"
  case "$name" in
      europarl-all|europarl)
	  KIND="parallel"
	  TARGET_DIR="$SHARDATA/europarl/all"
	  URLS=( "https://mammoth101.a3s.fi/europarl.tar.gz" )
	  EXTRACT_TGZ=1
	  ;;
      europarl-3langs)
	  KIND="parallel"
	  TARGET_DIR="$SHARDATA/europarl/3langs"
	  URLS=( "https://mammoth101.a3s.fi/europarl-3langs.tar.gz" )
	  EXTRACT_TGZ=1
	  ;;
      vocab-opusTC.mul)
	  KIND="vocab"
	  TARGET_DIR="$SHARDATA/vocab"
	  URLS=(
              "https://mammoth101.a3s.fi/opusTC.mul.64k.spm"
              "https://mammoth101.a3s.fi/opusTC.mul.vocab.onmt"
	  )
	  EXTRACT_TGZ=0
	  ;;
      tatoeba)
	  KIND="parallel"
	  TARGET_DIR="$SHARDATA/tatoeba/Tatoeba-Challenge"
	  URLS=("https://object.pouta.csc.fi/Tatoeba-Challenge-devtest/devtest.tar")
	  EXTRACT_TAR=0
	  # GIT_CLONE="https://github.com/Helsinki-NLP/Tatoeba-Challenge.git"
	  ;;
      opus100-de-en)
	  KIND="parallel"
	  TARGET_DIR="$SHARDATA/opus100/de-en"
	  URLS=( "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-de-en-v1.0.tar.gz" )
	  EXTRACT_TGZ=1
	  ;;
      opus100-zeroshot)
	  KIND="parallel"
	  TARGET_DIR="$SHARDATA/opus100/zeroshot"
	  URLS=( "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-zeroshot-v1.0.tar.gz" )
	  EXTRACT_TGZ=1
	  ;;
      *)
	  err "Unknown dataset: $name"; exit 2;;
  esac
}

# ─────────────────────────────── help & CLI ──────────────────────────────────
prog="${0##*/}"

list_datasets() {
  cat <<'DS'
Available datasets:
  europarl-3langs     → $SHARDATA/europarl/3langs
  europarl-all|europarl
                       → $SHARDATA/europarl/all
  vocab-opusTC.mul     → $SHARDATA/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
  tatoeba              → $SHARDATA/tatoeba/Tatoeba-Challenge (git)
  opus100-de-en        → $SHARDATA/opus100/de-en
  opus100-zeroshot     → $SHARDATA/opus100/zeroshot
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

Examples:
  SYSTEM=lumi  SHARDATA=/proj/xxx/shared $prog --check europarl-all
  SYSTEM=puhti SHARDATA=/proj/xxx/shared $prog --get   vocab-opusTC.mul
  SYSTEM=lumi  SHARDATA=/proj/xxx/shared $prog --stat  tatoeba

EOF
  list_datasets
}

# Parse args
ACTION=""; DATASET=""
case "${1:-}" in
  ""|-h|--help) usage; exit 0 ;;
  --list)       list_datasets; exit 0 ;;
  --check) ACTION="check"; DATASET="${2:-}"; shift 2 || true;;
  --get)   ACTION="get";   DATASET="${2:-}"; shift 2 || true;;
  --rm)    ACTION="rm";    DATASET="${2:-}"; shift 2 || true;;
  --stat)  ACTION="stat";  DATASET="${2:-}"; shift 2 || true;;
  *)
    # If user gave only a dataset name, default to check; otherwise it's an error
    if [[ -n "${1:-}" && -z "${2:-}" && "${1:-}" != --* ]]; then
      ACTION="check"; DATASET="$1"; shift
    else
      echo "❌ Unknown command or wrong arguments." >&2
      usage; exit 2
    fi
    ;;
esac

# Require dataset for actions that need one
if [[ -z "$DATASET" ]]; then
  echo "❌ Please specify a dataset." >&2
  usage; exit 2
fi

# Validate dataset name early (uses your existing pick_dataset)
if ! pick_dataset "$DATASET"; then
  echo "❌ Unknown dataset: $DATASET" >&2
  echo
  list_datasets
  exit 2
fi

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

extract_tgz(){
  local tgz="$1" to_dir="$2"
  ensure_dir "$to_dir"
  # idempotence: if target has content, assume extracted
  if compgen -G "$to_dir/*" >/dev/null; then
    log "looks extracted: $to_dir"
    return 0
  fi
  log "extracting $tgz -> $to_dir"
  tar -xzf "$tgz" -C "$to_dir"
}

extract_tar(){
  local tar="$1" to_dir="$2"
  ensure_dir "$to_dir"
  # idempotence: if target has content, assume extracted
  if compgen -G "$to_dir/*" >/dev/null; then
    log "looks extracted: $to_dir"
    return 0
  fi
  log "extracting $tar -> $to_dir"
  tar -xf "$tar" -C "$to_dir"
}

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

dataset_stats(){
  local dir="$1"
  if [[ ! -d "$dir" ]]; then echo "No data at $dir"; return 0; fi
  echo "== stats for $dir =="
  du -sh "$dir" || true
  find "$dir" -type f | wc -l | awk '{print "files:",$1}'
  # vocab special
  if [[ -f "$dir/opusTC.mul.vocab.onmt" ]]; then
    wc -l "$dir/opusTC.mul.vocab.onmt" | awk '{print "vocab lines:",$1}'
  fi
}

estimate_download_size(){
  local total=0 sz
  for u in "${URLS[@]:-}"; do
    sz=$(content_length "$u" || true)
    [[ -n "$sz" ]] && total=$(( total + sz ))
  done
  printf '%s' "$total"
}


# ─────────────────────────── actions (idempotent) ────────────────────────────
if [[ "$ACTION" == "check" ]]; then
    
  show_space_quota
  echo
  echo "== plan =="
  echo "dataset: $DATASET  kind: ${KIND:-git}"
  echo "target : $TARGET_DIR"
  
  if [[ -n "${GIT_CLONE:-}" ]]; then
    echo "source : $GIT_CLONE (git)"
    # no size estimate for git clone
    echo "size   : N/A (git repo; will depend on history)"
  else
    for u in "${URLS[@]}"; do echo "url    : $u"; done
    bytes=$(estimate_download_size || true)
    # Free space (bytes) on the filesystem hosting $SHARDATA
    avail_bytes=$(df -PB1 "$SHARDATA" | awk 'NR==2{print $4}')
    if [[ -n "${bytes:-}" && "$bytes" -gt 0 && "$avail_bytes" -lt "$bytes" ]]; then
	echo "❌ Not enough free space: need ~$(bytes_to_h "$bytes"), have $(bytes_to_h "$avail_bytes")" >&2
	exit 1
    fi
    if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
      echo "size   : $(bytes_to_h "$bytes")  ($(bytes_to_gib "$bytes") GiB)"
      print_storage_comparison "$bytes"
    else
      echo "size   : N/A (server did not send Content-Length)"
      # still show the LUMI line for comparison context
      lumi_tibh_month_line 0
    fi

    # ---- LUMI capacity-based recommendation (no TiB·h needed) ----
    if [[ "${SYSTEM,,}" == "lumi" && -n "${bytes:-}" && "$bytes" -gt 0 ]]; then
	echo
	echo "== capacity recommendation (from lumi-quota) =="
	if command -v lumi-quota >/dev/null 2>&1 && [[ -n "${ACCOUNT:-}" ]]; then
	    best_area=""; best_fit=0; chosen_bytes=0
	    while read -r area avail; do
		# pick the area with the largest headroom that still fits
		if (( avail > bytes )); then
		    if (( avail > best_fit )); then best_fit="$avail"; best_area="$area"; fi
		fi
		# remember the absolute biggest for a hint even if none fit
		(( avail > chosen_bytes )) && chosen_bytes="$avail"
		printf "  %-8s free: %8s (need ~%s)\n" "$area" "$(bytes_to_h "$avail")" "$(bytes_to_h "$bytes")"
	    done < <(get_lumi_quota_avail)
	    
	    if [[ -n "$best_area" ]]; then
		echo "✔ Recommend storing on: /$best_area (enough free space)."
	    else
		echo "❗ None of the areas have enough free capacity for this dataset."
		echo "   Largest free block is ~$(bytes_to_h "$chosen_bytes"). Consider reducing scope or contacting support."
	    fi
	else
	    echo "lumi-quota not available or ACCOUNT not set; set ACCOUNT and load lumi-tools:"
	    echo "  module load lumi-tools; export ACCOUNT=<project_account>"
	fi
    fi


    # ---- TiB·hour budget recommendation (based on the download plan) ----
    if [[ -n "$bytes" && "$bytes" -gt 0 ]]; then
	avail_tibh="$(get_available_tibh)"
	keep_tibh="$(tibh_for_bytes_days "$bytes" 30)"  # for ~30 days
	echo
	echo "== storage budget (TiB·hours) =="
	if [[ -n "$avail_tibh" ]]; then
	    echo "Project TiB·h (remaining): $avail_tibh"
	    echo "This dataset for 30 days : $keep_tibh TiB·h"
	    if awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{exit !(a<k)}'; then
		days_ok="$(days_supported_by_tibh "$avail_tibh" "$bytes")"
		echo "❗ Not enough budget for 30 days. You can keep it ~${days_ok} days with current TiB·h."
		echo
	    else
		# also estimate how many months fit
		months_fit=$(awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{printf "%.1f", a/k}')
		echo "✔ Fits in budget for ~${months_fit} month(s) at current size."
		echo
	    fi
	else
	    echo "Project TiB·h: unknown (set AVAILABLE_TIBH=... or ensure site tool is available)"
	    echo "This dataset for 30 days: $keep_tibh TiB·h"
	    echo
	fi
    fi
  fi
  echo
fi

case "$ACTION" in

    
  get)
    if [[ -n "${GIT_CLONE:-}" ]]; then
      clone_or_update "$GIT_CLONE" "$TARGET_DIR"
    else
      ensure_dir "$TARGET_DIR"
      tmp="$SHARDATA/.tmp-datasets"; ensure_dir "$tmp"
      for u in "${URLS[@]}"; do
        base="$(basename "$u")"
        part="$tmp/$base"
        download_file "$u" "$part"
        if (( EXTRACT_TGZ==1 )); then
          extract_tgz "$part" "$TARGET_DIR"
          # keep archives by default; rm -f "$part" if you want to clean
        elif (( EXTRACT_TAR==1 )); then
          extract_tar "$part" "$TARGET_DIR"
          # keep archives by default; rm -f "$part" if you want to clean
        else
          dest="$TARGET_DIR/$base"
          [[ -f "$dest" ]] || { mv "$part" "$dest"; log "moved $base -> $dest"; }
        fi
      done
      # sanity for vocab
      if [[ "$DATASET" = "vocab-opusTC.mul" ]]; then
        [[ -f "$TARGET_DIR/opusTC.mul.64k.spm" ]] || warn "missing .spm in $TARGET_DIR"
        [[ -f "$TARGET_DIR/opusTC.mul.vocab.onmt" ]] || warn "missing .onmt in $TARGET_DIR"
      fi
    fi
    log "done (--get $DATASET)"
    ;;


  
  rm)
    if [[ -e "$TARGET_DIR" ]]; then
      read -r -p "Remove $TARGET_DIR ? [y/N] " ans
      if [[ "$ans" =~ ^[Yy]$ ]]; then
        rm -rf "$TARGET_DIR"
        log "removed $TARGET_DIR"
      else
        log "aborted"
      fi
    else
      log "nothing to remove: $TARGET_DIR"
    fi
    ;;


  
  stat)
    dataset_stats "$TARGET_DIR"
    # precise bytes for BU/TiB-month & LUMI comparison
    if du -sb "$TARGET_DIR" >/dev/null 2>&1; then
      bytes_on_disk=$(du -sb "$TARGET_DIR" | awk '{print $1}')
      print_storage_comparison "$bytes_on_disk"

      # ---- LUMI capacity-based recommendation (no TiB·h needed) ----
      if [[ "${SYSTEM,,}" == "lumi" && -n "${bytes_on_disk:-}" && "$bytes_on_disk" -gt 0 ]]; then
	  echo
	  echo "== capacity recommendation (from lumi-quota) =="
	  if command -v lumi-quota >/dev/null 2>&1 && [[ -n "${ACCOUNT:-}" ]]; then
	      best_area=""; best_fit=0; chosen_bytes=0
	      while read -r area avail; do
		  # pick the area with the largest headroom that still fits
		  if (( avail > bytes_on_disk )); then
		      if (( avail > best_fit )); then best_fit="$avail"; best_area="$area"; fi
		  fi
		  # remember the absolute biggest for a hint even if none fit
		  (( avail > chosen_bytes )) && chosen_bytes="$avail"
		  printf "  %-8s free: %8s (need ~%s)\n" "$area" "$(bytes_to_h "$avail")" "$(bytes_to_h "$bytes_on_disk")"
	      done < <(get_lumi_quota_avail)

	      if [[ -n "$best_area" ]]; then
		  echo "✔ Recommend storing on: /$best_area (enough free space)."
	      else
		  echo "❗ None of the areas have enough free capacity for this dataset."
		  echo "   Largest free block is ~$(bytes_to_h "$chosen_bytes"). Consider reducing scope or contacting support."
	      fi
	  else
	      echo "lumi-quota not available or ACCOUNT not set; set ACCOUNT and load lumi-tools:"
	      echo "  module load lumi-tools; export ACCOUNT=<project_account>"
	  fi
      fi

      # ---- TiB·hour budget recommendation (based on on-disk size) ----
      if [[ -n "${bytes_on_disk:-}" && "$bytes_on_disk" -gt 0 ]]; then
	  
	  avail_tibh="$(get_available_tibh)"
	  keep_tibh="$(tibh_for_bytes_days "$bytes_on_disk" 30)"

	  echo
	  echo "== storage budget (TiB·hours) =="
	  if [[ -n "$avail_tibh" ]]; then
	      echo "Project TiB·h (remaining): $avail_tibh"
	      echo "Current data for 30 days : $keep_tibh TiB·h"
	      if awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{exit !(a<k)}'; then
		  days_ok="$(days_supported_by_tibh "$avail_tibh" "$bytes_on_disk")"
		  echo "❗ Not enough budget for 30 days. You can keep it ~${days_ok} days with current TiB·h."
		  echo
	      else
		  months_fit=$(awk -v a="$avail_tibh" -v k="$keep_tibh" 'BEGIN{printf "%.1f", a/k}')
		  echo "✔ Fits in budget for ~${months_fit} month(s) at current size."
		  echo
	      fi
	  else
	      echo "Project TiB·h: unknown (set AVAILABLE_TIBH=... or ensure site tool is available)"
	      echo "Current data for 30 days: $keep_tibh TiB·h"
	      echo
	  fi
      fi

    else
      echo "== storage comparison (≈30 days) =="
      echo "Size : N/A (du -sb not available on this system)"
      echo
      lumi_tibh_month_line 0
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




