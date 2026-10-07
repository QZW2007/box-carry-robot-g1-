#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Batch-run GVHMR over a folder of mp4s, then copy the results back to the
# Windows repo as data/<prefix>.pt.
#
#   bash gvhmr_batch.sh [mp4_dir] [out_root] [--fast] [--no-copy]
#
#   defaults : mp4_dir = auto-detected (the only dir under $HOME holding *.mp4)
#              out_root = ~/gvhmr_out
#   env vars : GVHMR_REPO  GVHMR repo root (auto-detected if unset)
#              PYTHON      interpreter to use; defaults to
#                          ~/miniconda3/envs/gvhmr/bin/python, so you do NOT
#                          need `conda activate` (this shell has no conda init)
#              WIN_DATA    destination data/ of the Windows repo
#
#   --fast     stop each job as soon as hmr4d_results.pt is complete, skipping
#              the two preview renders (demo.py saves the .pt BEFORE rendering)
#   --no-copy  do not copy the .pt files back to the Windows data/ folder
#
# Why not tools/demo/demo_folder.py: it is a thin `for` loop shelling out to
# demo.py once per video with check=True -> it re-loads the checkpoints for
# every clip and aborts the whole batch on the first failure.
# ---------------------------------------------------------------------------
set -o pipefail

MP4_DIR=""; OUT_ROOT=""; FAST=0; COPY=1
for a in "$@"; do
  case "$a" in
    --fast)    FAST=1 ;;
    --no-copy) COPY=0 ;;
    -h|--help) sed -n '2,/^set -o pipefail$/p' "$0"; exit 0 ;;
    -*)        echo "unknown option: $a" >&2; exit 2 ;;
    *)         if [ -z "$MP4_DIR" ]; then MP4_DIR="$a"
               elif [ -z "$OUT_ROOT" ]; then OUT_ROOT="$a"
               else echo "too many arguments" >&2; exit 2; fi ;;
  esac
done

for v in MP4_DIR OUT_ROOT; do
  eval "cur=\$$v"
  case "$cur" in "~"*) eval "$v=\"\${HOME}\${cur#\\~}\"" ;; esac
done
OUT_ROOT="${OUT_ROOT:-$HOME/gvhmr_out}"
WIN_DATA="${WIN_DATA:?set WIN_DATA to the Windows data dir, e.g. WIN_DATA=C:/path/to/repo/data}"

# --- locate the GVHMR repo (must contain tools/demo/demo.py) ---------------
REPO="${GVHMR_REPO:-}"
if [ -z "$REPO" ]; then
  for c in "$PWD" "$HOME/gvhmr" "$HOME/gvhmr/GVHMR" "$HOME/GVHMR" "$HOME/GVHMR-master"; do
    if [ -f "$c/tools/demo/demo.py" ]; then REPO="$c"; break; fi
  done
fi
if [ -z "$REPO" ] || [ ! -f "$REPO/tools/demo/demo.py" ]; then
  echo "[x] 找不到 GVHMR 仓库（需要有 tools/demo/demo.py）。"
  echo "    先进仓库根目录，或者： GVHMR_REPO=\"\$HOME/GVHMR\" bash $0"
  exit 1
fi

# --- find the mp4 folder if it was not given (or was mistyped) ------------
# Only descends 3 levels from $HOME and skips the heavy tool dirs, so it is
# instant even with a fat miniconda3 in the home directory.
scan_mp4_dirs() {
  local d f n
  for d in "$HOME"/*/ "$HOME"/*/*/ "$HOME"/*/*/*/; do
    [ -d "$d" ] || continue
    d="${d%/}"
    case "$d" in
      */miniconda3|*/miniconda3/*|*/.cache/*|*/.conda/*|*/.config/*|*/.nv/*|*/.git/*) continue ;;
      */node_modules/*|*/cuda_home/*|*/third_party/*|*/third-party/*) continue ;;
      */outputs|*/outputs/*) continue ;;   # GVHMR's own demo renders live here
    esac
    n=0
    for f in "$d"/*.mp4 "$d"/*.MP4; do
      [ -f "$f" ] && n=$((n + 1))
    done
    [ "$n" -gt 0 ] && echo "$n $d"
  done
}
if [ -n "$MP4_DIR" ] && [ ! -d "$MP4_DIR" ]; then
  echo "[!] 目录不存在: $MP4_DIR"
  MP4_DIR=""
fi
if [ -z "$MP4_DIR" ]; then
  CANDS=(); CNTS=()
  # most-mp4-first: a real recording batch has 10+ files, GVHMR's leftovers 1-2
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    CNTS+=("${line%% *}"); CANDS+=("${line#* }")
  done <<EOF
$(scan_mp4_dirs | sort -rn)
EOF
  if [ "${#CANDS[@]}" -eq 0 ]; then
    echo "[x] 在 \$HOME 下 3 层内没找到含 *.mp4 的目录。"
    echo "    请手动指定： bash $0 <mp4目录>"
    exit 1
  elif [ "${#CANDS[@]}" -eq 1 ] || [ "${CNTS[0]}" -gt "${CNTS[1]}" ]; then
    MP4_DIR="${CANDS[0]}"
    echo "[i] 自动选用 mp4 目录: $MP4_DIR （${CNTS[0]} 个 mp4）"
    i=1
    while [ "$i" -lt "${#CANDS[@]}" ]; do
      echo "      其他候选：${CNTS[$i]} 个  ${CANDS[$i]}"
      i=$((i + 1))
    done
  else
    echo "[x] 有多个同样多 mp4 的目录，请指定一个："
    i=0
    while [ "$i" -lt "${#CANDS[@]}" ]; do
      echo "      ${CNTS[$i]} 个  ${CANDS[$i]}"
      i=$((i + 1))
    done
    echo "    bash $0 <上面的某个目录>"
    exit 1
  fi
fi

# --- pick the interpreter: use the gvhmr env directly, no `conda activate` --
# The user's shell has no `conda init` block, so `conda activate` is not even on
# PATH there. Calling the env's python by absolute path is what has always
# worked, so do that by default (override with PYTHON=/path/to/python).
PYBIN="${PYTHON:-}"
if [ -z "$PYBIN" ]; then
  for c in "$HOME/miniconda3/envs/gvhmr/bin/python" \
           "$HOME/anaconda3/envs/gvhmr/bin/python" \
           "$HOME/miniconda3/envs/gvhmr/bin/python3"; do
    if [ -x "$c" ]; then PYBIN="$c"; break; fi
  done
fi
if [ -z "$PYBIN" ] && command -v conda >/dev/null 2>&1; then
  PYBIN="$(conda run -n gvhmr which python 2>/dev/null | tail -n 1)"
fi
PYBIN="${PYBIN:-python}"

if ! "$PYBIN" -c "import torch" >/dev/null 2>&1; then
  echo "[x] 这个 python 里没有 torch，GVHMR 跑不起来："
  echo "      $PYBIN"
  echo "    机器上装了这些 conda 环境："
  for c in "$HOME"/miniconda3/envs/*/bin/python "$HOME"/anaconda3/envs/*/bin/python; do
    [ -x "$c" ] && echo "      $c"
  done
  echo "    挑一个对的然后重跑： PYTHON=/上面的路径 bash $0 ${MP4_DIR:-<mp4目录>}"
  exit 1
fi
if ! "$PYBIN" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
  echo "[!] torch.cuda.is_available() == False —— 没拿到 GPU，GVHMR 会非常慢或直接失败。"
  echo
fi

mkdir -p "$OUT_ROOT" || exit 1

echo "repo     : $REPO"
echo "python   : $PYBIN"
echo "mp4 dir  : $MP4_DIR"
echo "out root : $OUT_ROOT"
echo "fast     : $FAST    (1 = 拿到 .pt 就停，跳过两个预览渲染)"
echo "copy back: $COPY    ($WIN_DATA)"
echo

cd "$REPO" || exit 1

# 0 = the file exists and torch can actually open it (catches a truncated .pt
# left behind by an interrupted/killed run).
verify_pt() {
  "$PYBIN" - "$1" <<'PY'
import sys
import torch
p = sys.argv[1]
try:
    torch.load(p, map_location="cpu", weights_only=False)
except TypeError:                      # torch < 2.0: no weights_only kwarg
    torch.load(p, map_location="cpu")
PY
}

OK=(); SKIP=(); FAIL=()

for f in "$MP4_DIR"/*.mp4 "$MP4_DIR"/*.MP4; do
  [ -e "$f" ] || continue
  p=$(basename "$f"); p="${p%.*}"
  dst="$OUT_ROOT/$p/hmr4d_results.pt"

  if [ -s "$dst" ] && verify_pt "$dst" >/dev/null 2>&1; then
    echo "[skip] $p   (结果已在且能读出)"
    SKIP+=("$p"); continue
  fi
  rm -f "$dst"                          # drop a possibly-truncated leftover

  echo "=== $p ==="
  if [ "$FAST" = 1 ]; then
    "$PYBIN" -m tools.demo.demo --video "$f" --output_root "$OUT_ROOT" -s & pid=$!
    stab=0; last=-1
    while kill -0 "$pid" 2>/dev/null; do
      s=$(stat -c%s "$dst" 2>/dev/null || echo -1)
      if [ "$s" != "-1" ] && [ "$s" = "$last" ]; then stab=$((stab+1)); else stab=0; fi
      last=$s
      if [ "$stab" -ge 3 ]; then kill "$pid" 2>/dev/null; break; fi
      sleep 3
    done
    wait "$pid" 2>/dev/null
  else
    "$PYBIN" -m tools.demo.demo --video "$f" --output_root "$OUT_ROOT" -s
  fi

  if [ -s "$dst" ] && verify_pt "$dst" >/dev/null 2>&1; then
    echo "[ok]   $p"
    OK+=("$p")
  else
    echo "[FAIL] $p   （这只会被跳过，其余继续）"
    FAIL+=("$p")
  fi
done

# --- copy the good ones back, renamed to the canonical prefix --------------
if [ "$COPY" = 1 ]; then
  if [ -d "$WIN_DATA" ]; then
    for p in ${OK[@]+"${OK[@]}"} ${SKIP[@]+"${SKIP[@]}"}; do
      cp -f "$OUT_ROOT/$p/hmr4d_results.pt" "$WIN_DATA/$p.pt" \
        && echo "[copy] data/$p.pt"
    done
  else
    echo "[!] 找不到目标 data/ 目录，跳过拷贝: $WIN_DATA"
    echo "    结果仍在 $OUT_ROOT/<前缀>/hmr4d_results.pt，可手动拷。"
  fi
fi

echo
echo "=============================== 汇总 ==============================="
echo "ok   (${#OK[@]}) : ${OK[*]:-}"
echo "skip (${#SKIP[@]}) : ${SKIP[*]:-}"
echo "FAIL (${#FAIL[@]}) : ${FAIL[*]:-}"
echo "===================================================================="
if [ "${#FAIL[@]}" -eq 0 ]; then
  echo "全部完成。回到 Windows 本机继续："
  echo '  python scripts\batch_pipeline.py --bag-root <数据根> --run'
else
  echo "有失败的条目 —— 直接重跑同一条命令即可（已完成的会自动跳过）："
  echo "  bash $0 $MP4_DIR"
  echo "或只补一条："
  echo "  cd $REPO && $PYBIN -m tools.demo.demo --video $MP4_DIR/<前缀>.mp4 --output_root $OUT_ROOT -s"
fi
