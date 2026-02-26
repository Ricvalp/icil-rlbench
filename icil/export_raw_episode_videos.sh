#!/usr/bin/env bash
set -euo pipefail

# Export one RGB video per variation for selected RLBench tasks.
#
# Default tasks:
#   - block_pyramid
#   - put_books_on_bookshelf
#
# Usage:
#   bash icil/export_raw_episode_videos.sh \
#     --raw-root /mnt/external_storage/robotics/rlbench/icil_rlbench \
#     --out-dir /mnt/external_storage/robotics/rlbench/raw_previews \
#     --num-variations 5

RAW_ROOT="${ICIL_RAW_ROOT:-/mnt/external_storage/robotics/rlbench/icil_rlbench}"
OUT_DIR="./raw_episode_videos"
NUM_VARIATIONS=3
CAMERA="front_rgb"
FPS=12
OVERWRITE=0

TASKS=(
  "block_pyramid"
  "put_books_on_bookshelf"
  "put_plate_in_colored_dish_rack"
  "stack_blocks"
  "take_money_out_safe"
)

usage() {
  cat <<'EOF'
Export one MP4 video per variation from RLBench raw episodes.

Options:
  --raw-root PATH        Root with task/variation*/episodes folders.
  --out-dir PATH         Output directory for generated mp4 files.
  --num-variations N     Max number of variations per task to export.
  --camera NAME          Camera folder name (default: front_rgb).
                         Common choices: front_rgb, wrist_rgb, overhead_rgb,
                         left_shoulder_rgb, right_shoulder_rgb
  --fps N                Input framerate for image sequence (default: 12).
  --overwrite            Overwrite existing output files.
  --tasks T1 T2 ...      Optional task list override.
  -h, --help             Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-root)
      RAW_ROOT="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --num-variations)
      NUM_VARIATIONS="$2"
      shift 2
      ;;
    --camera)
      CAMERA="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --tasks)
      shift
      TASKS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do
        TASKS+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found in PATH." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[export-videos] raw_root=$RAW_ROOT"
echo "[export-videos] out_dir=$OUT_DIR"
echo "[export-videos] tasks=${TASKS[*]}"
echo "[export-videos] num_variations=$NUM_VARIATIONS camera=$CAMERA fps=$FPS"

for task in "${TASKS[@]}"; do
  task_dir="$RAW_ROOT/$task"
  if [[ ! -d "$task_dir" ]]; then
    echo "[export-videos] skip missing task dir: $task_dir"
    continue
  fi

  mapfile -t var_dirs < <(find "$task_dir" -maxdepth 1 -mindepth 1 -type d -name 'variation*' | sort -V | head -n "$NUM_VARIATIONS")
  if [[ ${#var_dirs[@]} -eq 0 ]]; then
    echo "[export-videos] skip task with no variations: $task"
    continue
  fi

  for var_dir in "${var_dirs[@]}"; do
    var_name="$(basename "$var_dir")"
    episodes_dir="$var_dir/episodes"
    if [[ ! -d "$episodes_dir" ]]; then
      echo "[export-videos] skip missing episodes dir: $episodes_dir"
      continue
    fi

    mapfile -t episode_dirs < <(find "$episodes_dir" -maxdepth 1 -mindepth 1 -type d -name 'episode*' | sort -V)
    if [[ ${#episode_dirs[@]} -eq 0 ]]; then
      echo "[export-videos] skip $task/$var_name (no episodes)"
      continue
    fi

    ep_dir="${episode_dirs[0]}"
    ep_name="$(basename "$ep_dir")"
    rgb_dir="$ep_dir/$CAMERA"
    if [[ ! -d "$rgb_dir" ]]; then
      echo "[export-videos] skip missing camera dir: $rgb_dir"
      continue
    fi

    first_png="$rgb_dir/0.png"
    if [[ ! -f "$first_png" ]]; then
      echo "[export-videos] skip empty sequence (missing 0.png): $rgb_dir"
      continue
    fi

    out_task_dir="$OUT_DIR/$task"
    mkdir -p "$out_task_dir"
    out_mp4="$out_task_dir/${var_name}_${ep_name}_${CAMERA}.mp4"
    ffmpeg_overwrite_flag="-n"
    if [[ $OVERWRITE -eq 1 ]]; then
      ffmpeg_overwrite_flag="-y"
    fi

    echo "[export-videos] writing: $out_mp4"
    ffmpeg "$ffmpeg_overwrite_flag" \
      -framerate "$FPS" \
      -i "$rgb_dir/%d.png" \
      -c:v libx264 \
      -pix_fmt yuv420p \
      "$out_mp4" >/dev/null 2>&1 || {
        echo "[export-videos] ffmpeg failed for $rgb_dir" >&2
      }
  done
done

echo "[export-videos] done"
