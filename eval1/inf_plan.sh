#!/bin/bash
# inf_plan.sh
#
# Purpose
# -------
# Plan inference and evaluation commands for a set of test datasets.
#
# This script prepares the files needed for a batch evaluation cycle:
#   - per-task inference YAML files
#   - calls.out        (translation commands)
#   - calls.sacre.out  (SacreBLEU / ChrF commands)
#   - calls.comet.out  (COMET commands)
#
# It works by:
#   1) validating the required environment and directory structure,
#   2) building a language inventory from the internal triples table,
#   3) extracting and expanding task names from TRAINCONFIG,
#   4) checking that the expanded tasks are covered by known benchmark data,
#   5) generating per-task YAMLs via inf_extract_yaml.py,
#   6) writing translation and scoring command lists.
#
# Typical use
# -----------
# This script is normally called indirectly by the surrounding Makefile / shell
# workflow, not by hand. It expects all required environment variables to be set.
#
# Environment
# -----------
# Required:
#   DATADIR
#       Root directory of benchmark datasets.
#   OUTDIR
#       Output directory where per-task YAMLs and calls.* files are written.
#   MAMMOTH
#       Directory containing Mammoth runtime scripts, including translate.py.
#   MODEL
#       Model checkpoint or model directory used for inference.
#   TRAINCONFIG
#       Multi-task Mammoth training YAML.
#   BINDIR
#       Directory containing helper scripts such as inf_extract_yaml.py.
#   LOGDIR
#       Directory for inference stderr logs.
#   SCRDIR
#       Directory for metric outputs (.sacre, .comet).
#
# Reads
# -----
#   - TRAINCONFIG
#   - benchmark files under DATADIR
#   - BINDIR/inf_extract_yaml.py
#
# Writes
# ------
#   - OUTDIR/<task>.yaml              per-task inference YAML files
#   - OUTDIR/calls.out                translation commands
#   - OUTDIR/calls.sacre.out          SacreBLEU / ChrF commands
#   - OUTDIR/calls.comet.out          COMET commands
#   - OUTDIR/extraction_complete      marker file after YAML extraction
#
# Output
# ------
#   - planning status on stdout/stderr
#   - reusable command lists for inference and metric computation
#
# Notes
# -----
# - The script assumes Mammoth task names of shape:
#       mt_*, sentmt_*, docmt_*
# - Locale-aware task expansion is derived from the internal triples table.
# - Existing calls.out prevents regeneration, as a safety measure.

#!/bin/bash
# inf_plan.sh
#
# This is called by inf_make.sh.  Do not use directly if you do not set the environment.
#
# This script plans the inference tasks for test datasets.
# 1) Extract the tasks from the TRAINCONFIG and verify they are covered
# 2) Produce the inference calls based on the tasks we cover.
# 3) Checks that the resulting file  calls.out  contains only python commands
#
# Input
#   Variables: MAMMOTH, MODEL, TRAINCONFIG, OUTDIR, DATADIR
#   triples data structure with lines like  "fra fra_Latn fr_CA"
#   'mt_*' task names in TRAINCONFIG 
# Output
#   $OUTDIR/calls.out
# Caveat
#   assumes task names of shape   `mt_...`
#   fills the directory with task specific yaml files

echo "Starting inf_plan.sh..."

: "${DATADIR:?DATADIR must be set}"
: "${OUTDIR:?OUTDIR must be set}"
: "${MAMMOTH:?MAMMOTH must be set}"
: "${MODEL:?MODEL must be set}"
: "${TRAINCONFIG:?TRAINCONFIG must be set}"
: "${BINDIR:?BINDIR must be set}"
: "${LOGDIR:?LOGDIR must be set}"
: "${SCRDIR:?SCRDIR must be set}"
[[ -d "$MAMMOTH" ]] || { echo "Error: MAMMOTH must be an existing directory: $MAMMOTH" >&2; exit 1; }
[[ -e "$MODEL"   ]] || { echo "Error: MODEL must be an existing file or directory: $MODEL" >&2; exit 1; }
[[ -f "$TRAINCONFIG" ]] || { echo "Error: TRAINCONFIG must be an existing file: $TRAINCONFIG" >&2; exit 1; }
[[ -d "$OUTDIR" ]] || { echo "Error: OUTDIR must be an existing directory: $OUTDIR" >&2; exit 1; }
[[ -d "$DATADIR" ]] || { echo "Error: DATADIR must be an existing directory: $DATADIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_extract_yaml.py" ]] || { echo "Error: The inf_extract_yaml.py must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -d "$LOGDIR"                        ]] || { echo "Error: LOGDIR must be an existing directory: $LOGDIR" >&2; exit 1; }
[[ -d "$SCRDIR"                        ]] || { echo "Error: SCRDIR must be an existing directory: $SCRDIR" >&2; exit 1; }

triples=(
    "bos bos_Latn bs_??"
    "bul bul_Cyrl bg_BG"
    "cat cat_Latn ca_ES"
    "ces ces_Latn cs_CZ"
    "dan dan_Latn da_DK"
    "deu deu_Latn de_DE"
    "ell ell_Grek el_GR"
    "est ekk_Latn et_EE"
    "eng eng_Latn en_??"
    "eus eus_Latn eu_??"
    "fin fin_Latn fi_FI"
    "fra fra_Latn [fr_CA fr_FR]"
    "gle gle_Latn ga_??"
    "glg glg_Latn gl_??"
    "hrv hrv_Latn hr_HR"
    "hun hun_Latn hu_HU"
    "isl isl_Latn is_IS"
    "ita ita_Latn it_IT"
    "kat kat_Geor ka_??"
    "lav lvs_Latn lv_LV"
    "lit lit_Latn lt_LT"
    "mkd mkd_Cyrl mk_??"
    "mlt mlt_Latn mt_??"
    "nld nld_Latn nl_NL"
    "nno nno_Latn nn_??"
    "nob nob_Latn no_NO"
    "pol pol_Latn pl_PL"
    "por por_Latn [pt_PT pt_BR]"    # mv data/bouquet/test/por_Latn_braz1246.txt data/bouquet/test/por_Latn.txt  
    "ron ron_Latn ro_RO"
    "slk slk_Latn sk_SK"
    "slv slv_Latn sl_SI"
    "spa spa_Latn es_MX"
    "sqi als_Latn sq_??"
    "srp_Cyrl srp_Cyrl sr_RS"
    "swe swe_Latn sv_SE"
    "tur tur_Latn tr_TR"
    "ukr ukr_Cyrl uk_UA"
    # Add more lines for further languages
)
run_translation_if_input_exists() {
    local dataset="$1"
    local input="$2"
    local task="$3"
    local data="$6"
    local shortoutput="${task}.${data}.hyp"
    local refer="$4"
    local config="$5"
    local yamltask="$7"
    local input_lines output_lines
    echo "  $dataset" 
    if [ ! -e "$refer" ]; then
        echo "    FAIL - Reference does not   exist: $reference"
        return 0
    else
        echo "    FINE - Reference does       exist: $reference"
    fi
    if [ ! -e "$input" ]; then
        echo "    FAIL - Input does not       exist: $input"
        return 0
    else
        echo "    FINE - Input does           exist: $input"
    fi
    if [ ! -e "$config" ]; then
        echo "    FAIL - Config file does not exist: $config"
        exit 1
    else
        echo "    FINE - Config file does     exist: $config"	
    fi
    output="${OUTDIR}/$shortoutput"
    if [ -e "$output" ]; then
	#echo "counting lines..."
	input_lines=$(wc -l < "$input")
        output_lines=$(wc -l < "$output")
	#echo "done counting"
        if [ "$input_lines" -eq "$output_lines" ]; then
#	    if [ -e "${SCRDIR}/${TASK}.${data}.sacre" ] && [ "$(wc -l < "${SCRDIR}/${TASK}.${data}.sacre")" -eq 40 ]; then
	    if [ -s "${SCRDIR}/${TASK}.${data}.sacre" ] && [ "$(stat -c%s "${SCRDIR}/${TASK}.${data}.sacre")" -gt 500 ]; then
#	    if [ -s "${SCRDIR}/${TASK}.${data}.sacre" ]; then            
		echo "    GOOD - already scored: ${SCRDIR}/${TASK}.${data}.sacre"
	    else
		echo "    DONE - ($input_lines lines on both) -- ready to measure"
		echo "           Contrasting $refer  VS  $output"
		echo "if [ -s \"${SCRDIR}/${TASK}.${data}.sacre\" ] && [ \"\$(stat -c%s \"${SCRDIR}/${TASK}.${data}.sacre\")\" -gt 500 ]; then" >> "$OUTDIR/calls.sacre.out"
		echo "    echo \"Skipping ${SCRDIR}/${TASK}.${data}.sacre because it already exists and is >500 bytes\"" >> "$OUTDIR/calls.sacre.out"
		echo "else" >> "$OUTDIR/calls.sacre.out"
		echo "    date  | tee \"${SCRDIR}/${TASK}.${data}.sacre\""                                                    >> $OUTDIR/calls.sacre.out
		echo "    echo ${data} ${TASK} Contrasting $refer  VS  $output | tee -a \"${SCRDIR}/${TASK}.${data}.sacre\""  >> $OUTDIR/calls.sacre.out
		echo "    sacrebleu      $refer -i $output -m bleu chrf        | tee -a \"${SCRDIR}/${TASK}.${data}.sacre\""  >> $OUTDIR/calls.sacre.out
#		echo "    sacrebleu      $refer -i $output -m bleu chrf ter    | tee -a \"${SCRDIR}/${TASK}.${data}.sacre\""  >> $OUTDIR/calls.sacre.out
		echo "fi" >> "$OUTDIR/calls.sacre.out"
	    fi
	    echo "comet-score -r $refer -t $output -s $input --gpus 1  | tee \"${SCRDIR}/${TASK}.${data}.comet\""  >> $OUTDIR/calls.comet.out
            return 0
        else
            echo "    REDO - ($input_lines vs $output_lines lines; re-translate)  ==> $shortoutput"
        fi
    else
        echo "    TODO  - (translate now)  ==> $shortoutput"
    fi
    echo "    $input"
    echo "python -u \"${MAMMOTH}/translate.py\"" \
    "--config \"${config}\" --model \"${MODEL}\" --task_id \"${yamltask}\" --src \"${input}\"" \
    "--output \"${output}\" 2>>\"${LOGDIR}/job\${SLURM_JOB_ID}.${TASK}.err\"" >> $OUTDIR/calls.out
    SOME=1
}
echo "1) load cray-python and venv"
module load cray-python
source /scratch/project_462000964/shared/mammoth-shared/.venv/bin/activate
# source $PROJHOME/venvs/eval/bin/activate

echo "2) checking if calls.out already exists"
if [[ -s "$OUTDIR/calls.out" ]]; then
    echo "The file   calls.out   presumably contains inference calls already. I refuse to regenerate the plan."
    exit 0
fi

echo "3) building lookup tables of language codes and tasks"

# Build a lookup table of valid language codes from triples
# ----------------------------
# Helpers
# ----------------------------
# fr_FR -> FR
# fr_CA -> CA
# bg_BG -> XX   only if the language has no alternative variants (handled below)
# bs_?? -> XX
variant_tag_for_locale() {
    local locale="$1"
    local tag
    if [[ "$locale" == *"_"* ]]; then
        tag="${locale##*_}"
    else
        tag="XX"
    fi
    [[ -z "$tag" || "$tag" == "??" ]] && tag="XX"
    printf '%s\n' "$tag"
}

# If spec is already extended (e.g. FR.fra), return it as-is.
# Otherwise expand base language (e.g. fra) to all valid extended sides:
#   fra -> FR.fra CA.fra
#   bul -> XX.bul
expand_langspec() {
    local spec="$1"
    if [[ "$spec" == *.* ]]; then
        [[ -n "${valid_sides[$spec]:-}" ]] && printf '%s\n' "$spec"
        return
    fi
    [[ -z "${code2sides[$spec]:-}" ]] && return
    local side
    for side in ${code2sides[$spec]}; do
        printf '%s\n' "$side"
    done
}

# ----------------------------
# Build locale-aware language inventory from triples
# ----------------------------
declare -A code2ref          # fra -> fra_Latn
declare -A code2sides        # fra -> "FR.fra CA.fra", bul -> "XX.bul"
declare -A valid_codes       # fra -> 1
declare -A valid_sides       # FR.fra -> 1
declare -A side2code         # FR.fra -> fra
declare -A side2variant      # FR.fra -> FR
declare -A side2wmt          # FR.fra -> fr_FR ; XX.bul -> bg_BG

for triple in "${triples[@]}"; do
    read -r code ref wmt_rest <<< "$triple"
    valid_codes["$code"]=1
    code2ref["$code"]="$ref"
    # Strip optional brackets: [fr_CA fr_FR] -> fr_CA fr_FR
    wmt_rest="${wmt_rest#[}"
    wmt_rest="${wmt_rest%]}"
    read -r -a locales <<< "$wmt_rest"
    sides=()
    if (( ${#locales[@]} <= 1 )); then
        # Single-locale language: use XX.<code>
        locale="${locales[0]}"
        side="XX.${code}"
        valid_sides["$side"]=1
        side2code["$side"]="$code"
        side2variant["$side"]="XX"
        side2wmt["$side"]="$locale"
        sides+=("$side")
    else
        # Multi-locale language: one extended side per locale
        for locale in "${locales[@]}"; do
            variant="$(variant_tag_for_locale "$locale")"
            side="${variant}.${code}"
            valid_sides["$side"]=1
            side2code["$side"]="$code"
            side2variant["$side"]="$variant"
            side2wmt["$side"]="$locale"
            sides+=("$side")
        done
    fi
    code2sides["$code"]="${sides[*]}"
done

# ----------------------------
# Extract tasks from TRAINCONFIG and expand them
# ----------------------------
echo "4) Extract the tasks from the TRAINCONFIG and verify they are covered"
mapfile -t TASKS < <(yq r "$TRAINCONFIG" 'tasks.*' --printMode p | sed 's/^tasks[.]//')
declare -A task_set          # extended task names
declare -A greeted
declare -A task2yaml         # extended task -> yaml task key to consult later
for yaml_task in "${TASKS[@]}"; do
    # Match:
    #   mt_eng-fra
    #   mt_XX.eng-FR.fra
    #   sentmt_fra-eng
    #   docmt_FR.fra-XX.eng
    if [[ $yaml_task =~ ^(denoise)_([^-]+)-([^-]+)$ ]]; then
	continue
    elif [[ $yaml_task =~ ^(mt|sentmt|docmt)_([^-]+)-([^-]+)$ ]]; then
        prefix="${BASH_REMATCH[1]}"
        src_spec="${BASH_REMATCH[2]}"
        tgt_spec="${BASH_REMATCH[3]}"
        mapfile -t src_sides < <(expand_langspec "$src_spec")
        mapfile -t tgt_sides < <(expand_langspec "$tgt_spec")
        if (( ${#src_sides[@]} == 0 || ${#tgt_sides[@]} == 0 )); then
            echo "WARNING: could not expand YAML task '$yaml_task'" >&2
            continue
        fi
        for src_side in "${src_sides[@]}"; do
            for tgt_side in "${tgt_sides[@]}"; do
                ext_task="${prefix}_${src_side}-${tgt_side}"
                task_set["$ext_task"]=1
                task2yaml["$ext_task"]="$yaml_task"
		# echo "extended task name $yaml_task => $ext_task by adding a locale code (XX if only)"
            done
        done
    else
        echo "WARNING: unrecognized task format in YAML: $yaml_task" >&2
    fi
done
# ----------------------------
# Greet every expanded task whose two sides are valid
# ----------------------------
PRE=$'Following tasks do not have any test data:\n'
for task in "${!task_set[@]}"; do
    if [[ $task =~ ^(mt|sentmt|docmt)_([^-]+)-([^-]+)$ ]]; then
        src_side="${BASH_REMATCH[2]}"
        tgt_side="${BASH_REMATCH[3]}"
        if [[ -n "${valid_sides[$src_side]:-}" && -n "${valid_sides[$tgt_side]:-}" ]]; then
            greeted["$task"]=1
#           echo -n "$task "
        else
	    printf '%s' "$PRE"
            printf '%s\n' "$task"
            PRE=""
	fi
    fi
done
echo
if [[ -z $PRE ]]; then
    echo "FAIL - I refuse to run before all tasks are covered (or the script is generalized)"
    exit 1
fi



echo "5) Produce the inference calls based on the tasks we cover; does not redo unnecessarily."
if [[ -e "$OUTDIR/calls.out" ]]; then
    echo "The file   calls.out   presumably contains inference calls already. I refuse to regenerate the plan."
    exit 0
else
    echo "Producing the yaml files..."
    if [ ! -e "${OUTDIR}/extraction_complete" ]; then 
	python "$BINDIR/inf_extract_yaml.py" "${TRAINCONFIG}" "${OUTDIR}" 
	touch "${OUTDIR}/extraction_complete"
    fi
    
    echo "Producing the inference calls ..."
    echo -n "" >$OUTDIR/calls.out
    echo -n "" >$OUTDIR/calls.comet.out
    echo -n "" >$OUTDIR/calls.sacre.out

    for TASK in "${!task_set[@]}"; do
	# Prefer the original YAML task key if planner expansion created this TASK from it
	YAMLTASK="${task2yaml[$TASK]:-$TASK}"
	TRANSLATECONFIG="${OUTDIR}/${YAMLTASK}.yaml"
	# 1) we still use one task per config file - unoptimal
	# 2) next step: all mt task present, but tell with --task which is inferred
	# 3) then: use original train.yaml, but tell with --task which is inferred
	echo " "
	echo "$TASK:  "
	# Match task names like:
	#   mt_eng-bos mt_bos-eng mt_bos-bul sentmt_bos-swe docmt_eng-srp_Cyrl
	#   mt_FR.fra-XX.eng docmt_XX.srp_Cyrl-FR.fra
	if [[ $TASK =~ ^(mt|sentmt|docmt)_([^-]+)-([^-]+)$ ]]; then
            TASKTYPE="${BASH_REMATCH[1]}"
            SRC_SIDE="${BASH_REMATCH[2]}"
            TGT_SIDE="${BASH_REMATCH[3]}"
            # Split SRC/TGT sides: VARIANT.CODE or plain CODE
            if [[ "$SRC_SIDE" == *.* ]]; then
		SRC_VARIANT="${SRC_SIDE%%.*}"
		SRC="${SRC_SIDE#*.}"
            else
		SRC_VARIANT="XX"
		SRC="$SRC_SIDE"
            fi
            if [[ "$TGT_SIDE" == *.* ]]; then
		TGT_VARIANT="${TGT_SIDE%%.*}"
		TGT="${TGT_SIDE#*.}"
            else
		TGT_VARIANT="XX"
		TGT="$TGT_SIDE"
            fi
            # Both base languages must be known
            if [[ -z "${code2ref[$SRC]:-}" || -z "${code2ref[$TGT]:-}" ]]; then
		echo "  Skipping $TASK: unknown language code(s)"
		continue
            fi
            SRC_REF="${code2ref[$SRC]}"
            TGT_REF="${code2ref[$TGT]}"
            # For WMT, prefer side-specific mapping if available; otherwise fall back to code2wmt
            SRC_WMT="${side2wmt[$SRC_SIDE]:-${code2wmt[$SRC]:-}}"
            TGT_WMT="${side2wmt[$TGT_SIDE]:-${code2wmt[$TGT]:-}}"    
            # Flores+ and BOUQuET always come from the base-language pool
            INPUT1="$DATADIR/flores_plus/devtest/${SRC_REF}.txt"
            REFER1="$DATADIR/flores_plus/devtest/${TGT_REF}.txt"
            INPUT2="$DATADIR/bouquet/test/${SRC_REF}.txt"
            REFER2="$DATADIR/bouquet/test/${TGT_REF}.txt"   
            INPUT4="$DATADIR/bouquet_par/test/${SRC_REF}.txt"
            REFER4="$DATADIR/bouquet_par/test/${TGT_REF}.txt"   
            # WMT24++ handling:
            if [[ "$SRC" == "eng" ]]; then
		# eng -> X
		INPUT3="$DATADIR/wmt24pp/train/en-${TGT_WMT}/source.en.txt"
		REFER3="$DATADIR/wmt24pp/train/en-${TGT_WMT}/reference.target.txt"
            elif [[ "$TGT" == "eng" ]]; then
		# X -> eng
		INPUT3="$DATADIR/wmt24pp/train/en-${SRC_WMT}/reference.target.txt"
		REFER3="$DATADIR/wmt24pp/train/en-${SRC_WMT}/source.en.txt"
            else
		# X -> Y, both non-English
		INPUT3="$DATADIR/wmt24pp/train/en-${SRC_WMT}/reference.target.txt"
		REFER3="$DATADIR/wmt24pp/train/en-${TGT_WMT}/reference.target.txt"
            fi
            run_translation_if_input_exists "Flores+" "$INPUT1" "$TASK" "$REFER1" ${TRANSLATECONFIG} "flo" "$YAMLTASK"
            run_translation_if_input_exists "BOUQuET" "$INPUT2" "$TASK" "$REFER2" ${TRANSLATECONFIG} "bqt" "$YAMLTASK"
            run_translation_if_input_exists "WMT24++" "$INPUT3" "$TASK" "$REFER3" ${TRANSLATECONFIG} "wmt" "$YAMLTASK"
            run_translation_if_input_exists "BOUQuETpar" "$INPUT4" "$TASK" "$REFER4" ${TRANSLATECONFIG} "bqtpar" "$YAMLTASK"
	fi
#	if [[ -n $SOME ]]; then
#	    exit 0
#	fi
    done
fi

echo "6) Check that all calls refer to python...."
if grep -qv '^python' $OUTDIR/calls.out; then
    echo "FAIL: not every line starts with 'python'"
    grep -nv '^python' $OUTDIR/calls.out
    exit 1
else
    echo "OK: every line starts with 'python'"
fi
