#!/bin/bash -f
# ======= Hoffman2 SGE headers =======
#$ -cwd
#$ -o $JOB_ID.log
#$ -e $JOB_ID.err
# 申请并行环境（按你平时用的 dc* 队列）
#$ -pe dc* 32
# 资源与时限（可调）
#$ -l h_data=4G,h_vmem=16G,h_rt=24:00:00
#$ -V

# ======= Modules / env =======
source /u/local/Modules/default/init/modules.sh
module unload intel/2020.4 2>/dev/null || true
module load intel/17.0.7

# VASP 赝势目录（按你提供的）
export VASP_PP_PATH="$HOME/vasp/mypps"

# MPI/线程
export OMP_NUM_THREADS=1
export I_MPI_COMPATIBILITY=4

# ======= Guardrails: inputs =======
# POSCAR 兜底：优先 slab.POSCAR，其次 structure.POSCAR
if [ ! -f POSCAR ]; then
  if   [ -f slab.POSCAR ]; then cp slab.POSCAR POSCAR
  elif [ -f structure.POSCAR ]; then cp structure.POSCAR POSCAR
  else
    echo "[ERROR] POSCAR not found, and no slab.POSCAR/structure.POSCAR to fallback."
    exit 2
  fi
fi

# POTCAR 兜底：用 POTCAR.spec + $VASP_PP_PATH 组装（支持 potpaw_* 布局）
if [ ! -f POTCAR ] && [ -f POTCAR.spec ]; then
  echo "[SGE] Assembling POTCAR from \$VASP_PP_PATH=$VASP_PP_PATH"

  python3 - <<'PY'
import json, os, sys, glob

root = os.path.expanduser(os.environ.get("VASP_PP_PATH",""))
if not os.path.isdir(root):
    print(f"[ERROR] VASP_PP_PATH not a directory: {root}", file=sys.stderr); sys.exit(2)

# 读取 spec：支持 ["Cu","O"] 或 {"Cu":"potpaw_PBE/Cu/POTCAR", ...}
try:
    spec = json.load(open("POTCAR.spec"))
except Exception as e:
    print(f"[ERROR] read POTCAR.spec: {e}", file=sys.stderr); sys.exit(2)

pairs = []
if isinstance(spec, list):
    pairs = [(str(x), None) for x in spec]
elif isinstance(spec, dict):
    pairs = [(k, spec.get(k)) for k in spec]
else:
    print("[ERROR] POTCAR.spec must be list or dict", file=sys.stderr); sys.exit(2)

def first_existing(rel_list):
    for rel in rel_list:
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None

open("POTCAR","wb").close()
okcnt = 0
for elem, hint in pairs:
    candidates = []
    if isinstance(hint, str) and hint.strip():
        candidates.append(hint.strip())
    # 常见布局（包含 potpaw_*）
    candidates += [
        f"potpaw_PBE/{elem}/POTCAR",
        f"potpaw_GGA/{elem}/POTCAR",
        f"potpaw/{elem}/POTCAR",
        f"{elem}/POTCAR",
        f"{elem}_pv/POTCAR",
    ]
    # 单文件命名也试一下
    candidates += [f"POTCAR_{elem}", f"POTCAR.{elem}", f"POTCAR-{elem}"]

    found = first_existing(candidates)
    if not found:
        # 兜底全局搜（截 3 层深度，且路径里包含元素名）
        hits = [p for p in glob.glob(os.path.join(root, "**", "POTCAR"), recursive=True)
                if (f"/{elem}/" in p or f"/{elem}_pv/" in p)]
        if hits:
            found = hits[0]

    if not found:
        print(f"[WARN] POTCAR for {elem} not found under {root}", file=sys.stderr)
        continue

    with open("POTCAR","ab") as out, open(found,"rb") as f:
        out.write(f.read())
    okcnt += 1
    print(f"[OK] add {elem} from {os.path.relpath(found, root)}")

if okcnt == 0:
    print("[ERROR] No POTCAR pieces assembled.", file=sys.stderr); sys.exit(3)
PY

  if [ $? -ne 0 ]; then
    echo "[ERROR] POTCAR assembly failed" >&2
    exit 3
  fi
fi

# 基本输入存在性检查
for f in INCAR KPOINTS POSCAR; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing $f" >&2
    exit 3
  fi
done

# ======= Run VASP =======
echo "[INFO] Starting VASP @ $(date) on $(hostname)"
echo "[INFO] NSLOTS=${NSLOTS:-1}"
# 你的 vasp 可执行文件
VASP_BIN="$HOME/vasp_std_vtst_sol"

# 可选：把作业环境记录下来
env | sort > _env.dump

# 真正运行
mpirun -np ${NSLOTS:-1} "$VASP_BIN" > vasp.stdout 2> vasp.stderr
rc=$?

echo "run complete on $(hostname): $(date) $(pwd)" >> $HOME/job.log
echo "[INFO] Done with rc=$rc"
exit $rc