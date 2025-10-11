#!/usr/bin/env bash
set -euo pipefail

# This script downloads and organizes datasets needed by data/prepare_grounding.py
# Target layout (under vg_new/):
#   vg_new/
#     images/VG_100K/           # Visual Genome images part 1
#     images2/VG_100K_2/        # Visual Genome images part 2
#     coco/train2014/           # MSCOCO 2014 training images
#     coco/val2014/             # MSCOCO 2014 validation images
#     coco/annotations/         # COCO 2014 JSONs (we reuse repo's coco_annotations)
#     flickr30k/                # Flickr30k images
#
# Notes:
# - This script will extract local archives if available (e.g., data/vg/flickr30k-*.tar.gz)
# - For large downloads (VG/COCO), it uses public mirrors. Feel free to replace URLs with faster mirrors.

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)
TARGET_DIR="${ROOT_DIR}/vg_new"
COCO_ANN_SRC="${ROOT_DIR}/coco_annotations"

# Helpers
has_content() {
  local d="$1"
  [[ -d "$d" ]] && [[ -n "$(ls -A "$d" 2>/dev/null || true)" ]]
}

link_if_exists() {
  local src="$1"; local dst="$2"
  if [[ -d "$src" ]] && has_content "$src"; then
    echo "Linking existing: $src -> $dst"
    if [[ -d "$dst" && ! -L "$dst" && -z "$(ls -A "$dst" 2>/dev/null || true)" ]]; then
      rmdir "$dst"
    fi
    if [[ -e "$dst" && ! -L "$dst" ]]; then
      echo "WARNING: $dst exists and is not a symlink; skipping link."
    else
      ln -sfn "$src" "$dst"
    fi
    return 0
  fi
  return 1
}

mkdir -p "${TARGET_DIR}/images/VG_100K" "${TARGET_DIR}/images2/VG_100K_2" \
         "${TARGET_DIR}/coco/train2014" "${TARGET_DIR}/coco/val2014" "${TARGET_DIR}/coco/annotations" \
         "${TARGET_DIR}/flickr30k" "${TARGET_DIR}/annotations"

printf "\n==> Preparing Flickr30k from local archives or existing dirs...\n"
# Prefer existing directories in root or vg
link_if_exists "${ROOT_DIR}/flickr30k" "${TARGET_DIR}/flickr30k" || \
link_if_exists "${ROOT_DIR}/data/flickr30k" "${TARGET_DIR}/flickr30k" || true

# Try to extract flickr30k archives if present in repo
if [ -f "${ROOT_DIR}/data/vg/flickr30k-images.tar.gz" ]; then
  echo "Extracting flickr30k-images.tar.gz ..."
  tar -xzf "${ROOT_DIR}/data/vg/flickr30k-images.tar.gz" -C "${TARGET_DIR}/flickr30k"
fi
if [ -f "${ROOT_DIR}/data/vg/flickr30k.tar.gz" ]; then
  echo "Extracting flickr30k.tar.gz ..."
  tar -xzf "${ROOT_DIR}/data/vg/flickr30k.tar.gz" -C "${TARGET_DIR}/flickr30k"
fi

printf "\n==> Linking COCO annotations (using repo's coco_annotations)...\n"
if [ -d "${COCO_ANN_SRC}" ]; then
  # Create symlinks for known annotation files
  for f in captions_train2014.json captions_val2014.json instances_train2014.json instances_val2014.json person_keypoints_train2014.json person_keypoints_val2014.json; do
    if [ -f "${COCO_ANN_SRC}/$f" ]; then
      ln -sfn "${COCO_ANN_SRC}/$f" "${TARGET_DIR}/coco/annotations/$f"
    fi
  done
else
  echo "WARNING: coco_annotations directory not found at ${COCO_ANN_SRC}. Skipping annotation links."
fi

printf "\n==> Linking RefCOCO-family annotations if available...\n"
# These are expected under data/vg_/ from Shikra-style datasets
VG_ANN_ROOT="${ROOT_DIR}/data/vg_"
if [ -d "${VG_ANN_ROOT}" ]; then
  for d in refclef refcoco "refcoco+" refcocog; do
    src="${VG_ANN_ROOT}/$d"
    dst="${TARGET_DIR}/annotations/$d"
    if [[ -d "$src" ]] && has_content "$src"; then
      echo "Linking $d annotations: $src -> $dst"
      ln -sfn "$src" "$dst"
    else
      echo "Note: $d annotations not found at $src, skipping."
    fi
  done
  # Optional: link Flickr30k bottom-up features if present (may be useful for some pipelines)
  for d in flickrbu_att flickrbu_box flickrbu_fc; do
    src="${VG_ANN_ROOT}/$d"
    dst="${TARGET_DIR}/$d"
    if [[ -d "$src" ]] && has_content "$src"; then
      echo "Linking Flickr feature folder: $src -> $dst"
      ln -sfn "$src" "$dst"
    fi
  done
else
  echo "NOTE: ${VG_ANN_ROOT} not found; skipping RefCOCO-family annotation linking."
fi

printf "\n==> Visual Genome images (VG_100K & VG_100K_2): link local if available, otherwise download...\n"
# Link existing VG folders if present under root or vg/
link_if_exists "${ROOT_DIR}/VG_100K" "${TARGET_DIR}/images/VG_100K" || \
link_if_exists "${ROOT_DIR}/vg/VG_100K" "${TARGET_DIR}/images/VG_100K" || true
link_if_exists "${ROOT_DIR}/VG_100K_2" "${TARGET_DIR}/images2/VG_100K_2" || \
link_if_exists "${ROOT_DIR}/vg/VG_100K_2" "${TARGET_DIR}/images2/VG_100K_2" || true
VG1_URLS=(
  "https://visualgenome.org/static/data/dataset/VG_100K.zip"
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip"  # alt mirror sometimes contains VG_100K
)
VG2_URLS=(
  "https://visualgenome.org/static/data/dataset/VG_100K_2.zip"
  "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip"
)

pushd "${TARGET_DIR}" >/dev/null
if ! has_content "${TARGET_DIR}/images/VG_100K"; then
  for u in "${VG1_URLS[@]}"; do
    echo "Trying $u ..."
    if wget -q --show-progress "$u" -O VG_100K.zip; then
      unzip -q VG_100K.zip -d images/ || true
      rm -f VG_100K.zip
      break
    fi
  done
fi
if ! has_content "${TARGET_DIR}/images2/VG_100K_2"; then
  for u in "${VG2_URLS[@]}"; do
    echo "Trying $u ..."
    if wget -q --show-progress "$u" -O VG_100K_2.zip; then
      unzip -q VG_100K_2.zip -d images2/ || true
      rm -f VG_100K_2.zip
      break
    fi
  done
fi
popd >/dev/null

printf "\n==> MSCOCO 2014 images: link local if available, otherwise download...\n"
# Link existing COCO train/val if present under common locations
link_if_exists "${ROOT_DIR}/coco/train2014" "${TARGET_DIR}/coco/train2014" || \
link_if_exists "${ROOT_DIR}/train2014" "${TARGET_DIR}/coco/train2014" || \
link_if_exists "${ROOT_DIR}/data/coco/train2014" "${TARGET_DIR}/coco/train2014" || true
link_if_exists "${ROOT_DIR}/coco/val2014" "${TARGET_DIR}/coco/val2014" || \
link_if_exists "${ROOT_DIR}/val2014" "${TARGET_DIR}/coco/val2014" || \
link_if_exists "${ROOT_DIR}/data/coco/val2014" "${TARGET_DIR}/coco/val2014" || true
COCO_BASE="http://images.cocodataset.org/zips"
COCO_TRAIN_ZIP="train2014.zip"
COCO_VAL_ZIP="val2014.zip"

if ! has_content "${TARGET_DIR}/coco/train2014"; then
  wget -c "${COCO_BASE}/${COCO_TRAIN_ZIP}" -O "${TARGET_DIR}/coco/${COCO_TRAIN_ZIP}"
  (cd "${TARGET_DIR}/coco" && unzip -q "${COCO_TRAIN_ZIP}" && rm -f "${COCO_TRAIN_ZIP}")
fi
if ! has_content "${TARGET_DIR}/coco/val2014"; then
  wget -c "${COCO_BASE}/${COCO_VAL_ZIP}" -O "${TARGET_DIR}/coco/${COCO_VAL_ZIP}"
  (cd "${TARGET_DIR}/coco" && unzip -q "${COCO_VAL_ZIP}" && rm -f "${COCO_VAL_ZIP}")
fi

printf "\nAll done. Dataset directories prepared under: ${TARGET_DIR}\n"