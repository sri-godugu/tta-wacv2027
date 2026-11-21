#!/bin/bash
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --requeue
# BN-statistics control worker (P7a/P7b; preregistration_step_curve.md addendum 2026-10-19 05:20 EDT).
# Copy of stepcurve_worker.sh (October campaign, unchanged) with ONE difference: evaluate.py is run through
# scripts/eval_lr0_identity.py, which records the registered bit-identity check of the parameters during the
# scored pass (results/stepcurve/$POOL/identity/<tag>.json). The evaluation flags are identical.
# Usage: stepcurve_worker.sh METHOD STEPS LR TARGET POOL
#   METHOD in {van,pda,tent,ora}  (van = no adaptation; STEPS/LR ignored, tag "van")
#   TARGET = "real" (8 real conditions, sequential), a Cityscapes-C v2 type (all 5
#            severities), or "type@sev" (one severity, for finer parallelism)
#   POOL   = GPU pool tag (h200 | v100). Each pool writes only to results/stepcurve/$POOL,
#            and benefit is always computed against the vanilla run of the same pool.
# Config is byte-matched to the published s10_* runs; only --n_adapt_steps / --lr_pda vary.
# Each output is written under an exclusive flock and skipped if it already exists, so a
# requeued job resumes cleanly and two concurrent jobs can never write the same file.
# NOTE: no `set -u` -- conda's activate/deactivate hooks reference unbound vars.
METHOD=$1; STEPS=$2; LR=$3; TARGET=$4; POOL=$5
[ -n "$POOL" ] || { echo "POOL required"; exit 2; }
[ "$METHOD" = pda ] && [ "$STEPS" = 1 ] && [ "$LR" = 0 ] || { echo "this worker is only for pda 1 0"; exit 2; }
R=${ROBUSTSEG_ROOT:-$(pwd)}   # repository root
OUT=$R/results/stepcurve/$POOL; mkdir -p $OUT/per_image_cm $OUT/identity
conda activate robustseg && cd $R/code   # any env with code/requirements.txt
# Allocator-only setting (no numeric effect): PDA peaks near a 24 GB card's limit.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case $METHOD in
  van)  MFLAGS="";                             TAGP="van" ;;
  pda)  MFLAGS="--use_pda --K 4 --clean_final"; TAGP="pda_s${STEPS}_lr${LR}" ;;
  tent) MFLAGS="--use_tent";                   TAGP="tent_s${STEPS}_lr${LR}" ;;
  ora)  MFLAGS="--use_oracle";                 TAGP="ora_s${STEPS}_lr${LR}" ;;
  *) echo "bad method $METHOD"; exit 2 ;;
esac
[ "$METHOD" = van ] && STEPS=1 && LR=0.001   # unused without an adaptation flag
BASE="python scripts/eval_lr0_identity.py --checkpoint $R/runs/tcpr_deeplab_pdaonly/best.pth --pretrained \
  --backbone deeplabv3plus --batch_size 2 --device cuda --seed 0 --texture_threshold 0.45 \
  --norm_types bn --n_adapt_steps $STEPS --lr_pda $LR $MFLAGS"
echo "[worker] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) pool=$POOL $TAGP $TARGET"

DS=$R/datasets
ACDC="--dataset acdc --data_root $DS/acdc --split val"
BDD="--dataset bdd100k --data_root $DS/bdd100k --split val --attr_json $DS/bdd100k/bdd_segval_attrs.json"
FZ="--dataset foggy_zurich --data_root $DS/foggy_zurich/Foggy_Zurich --condition all --split val"

bad_result () {  # an empty dataset or a crash leaves no file or an all-zero mIoU; never keep those
  [ ! -s "$1" ] && return 0
  grep -qE '"miou": 0(\.0+)?[,}]' "$1" && { rm -f "$1"; return 0; }
  return 1
}

run_locked () {  # json, label, ready-path (dir or file that must exist), eval args...
  local js=$1 label=$2 dir=$3; shift 3
  [ -s "$js" ] && { echo "[skip] $label"; return; }
  [ -e "$dir" ] || { echo "[MISSING DATA] $label: $dir"; return; }
  exec 9>"$js.lock"
  flock -n 9 || { echo "[locked-by-other] $label"; exec 9>&-; return; }
  if [ -s "$js" ]; then echo "[skip] $label"; else
    local t0=$(date +%s)
    $BASE --identity_json $OUT/identity/$(basename $js) "$@" --results_json $js 2>&1 | grep -E "^mIoU|^[1-5] |^Mean|OutOfMemory|Error|^IDENTITY"
    bad_result $js && echo "[FAILED] $label" || echo "[done] $label $(( $(date +%s) - t0 ))s"
  fi
  exec 9>&-   # lock file is left in place: unlinking a held lock reopens the race
}

real_one () {  # tag, data dir, dataset flags...
  local tag=$1 dir=$2; shift 2
  local strip=""   # qualitative strip: pre-registered ACDC-fog indices, saved from the scored pass
  [ "$tag" = acdc_fog ] && strip="--save_preds $OUT/strip_${TAGP}_acdc_fog.npz --save_pred_idx 14,42,51"
  run_locked $OUT/${TAGP}_${tag}.json ${TAGP}_${tag} $dir "$@" \
    --per_image_cm $OUT/per_image_cm/${TAGP}_${tag}.npz $strip
}

if [ "$TARGET" = "real" ]; then
  for c in fog night rain snow; do real_one acdc_$c $DS/acdc/rgb_anon/$c/val $ACDC --condition $c; done
  real_one bdd_rainy $DS/bdd100k/images $BDD --condition rainy --drop_last
  real_one bdd_snowy $DS/bdd100k/images $BDD --condition snowy
  real_one bdd_clear $DS/bdd100k/images $BDD --condition clear
  real_one fz_fog    $DS/foggy_zurich/Foggy_Zurich $FZ
else
  # Ready only once that type's generation + determinism self-check finished (manifest is written last).
  TYPE=${TARGET%@*}; SEV=0; SFX=""
  [[ "$TARGET" == *@* ]] && { SEV=${TARGET#*@}; SFX="_sev$SEV"; }
  run_locked $OUT/${TAGP}_v2pertype_${TYPE}${SFX}.json ${TAGP}_v2pertype_${TYPE}${SFX} \
    $DS/cityscapes_c_v2/MANIFEST_seeds_${TYPE}.csv \
    --dataset cityscapes_c --data_root $DS/cityscapes_c_v2 --cs_root $DS/cityscapes \
    --corruption_type $TYPE --severity $SEV
fi
