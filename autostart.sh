#!/usr/bin/env bash
# ============================================================================
# 2027_preview 打符程序自启动脚本
#
#   用法:  ./autostart.sh [配置文件] [程序名]
#   例:    ./autostart.sh                                   # configs/sentry.yaml + auto_buff_debug_mpc
#          ./autostart.sh configs/standard3.yaml
#          ./autostart.sh configs/sentry.yaml standard_mpc
#          DRY_RUN=1 ./autostart.sh configs/sentry.yaml     # 只打印将要执行的命令，不真的启动
#
# 相比旧版的改动:
#   1) 旧版写死 `cd ~/Desktop/sp_vision_25/`，换机器/换目录必然起不来；
#      现在用脚本自身位置定位仓库根目录。
#   2) 旧版依赖 `./watchdog.sh`，而该文件在 .gitignore 里且仓库中并不存在；
#      现在由本脚本自己生成守护循环（进程崩了自动重启）。
#   3) 启动前校验可执行文件、配置文件、com_port 是否存在，失败给出明确提示。
# ============================================================================
set -u

CONFIG="${1:-configs/sentry.yaml}"
PROGRAM="${2:-auto_buff_debug_mpc}"
DRY_RUN="${DRY_RUN:-0}"
SCREEN_NAME="buff"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || { echo "[autostart] 无法进入 $REPO_DIR"; exit 1; }

BIN="$REPO_DIR/build/$PROGRAM"
if [ ! -x "$BIN" ]; then
  echo "[autostart] 找不到可执行文件: $BIN"
  echo "           先编译: cmake -B build && make -C build -j\$(nproc)"
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "[autostart] 找不到配置文件: $CONFIG"
  exit 1
fi

mkdir -p logs
STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUNNER="logs/${STAMP}.runner.sh"

# io::Gimbal 从配置里读 com_port，缺了会直接 exit(1)，这里提前提示
COM_PORT="$(sed -n 's/^com_port:[[:space:]]*"\?\([^"# ]*\)"\?.*/\1/p' "$CONFIG" | head -1)"

echo "[autostart] 仓库目录 = $REPO_DIR"
echo "[autostart] 程序     = $PROGRAM"
echo "[autostart] 配置     = $CONFIG"
echo "[autostart] 串口     = ${COM_PORT:-（配置里没有 com_port）}"

if [ -z "$COM_PORT" ]; then
  echo "[autostart] 警告: $CONFIG 里没有 com_port，io::Gimbal 会启动失败。"
elif [ ! -e "$COM_PORT" ]; then
  echo "[autostart] 警告: 串口 $COM_PORT 不存在，请检查 udev 规则 / 接线 / 是否插好。"
fi

# 守护循环：崩了就重启，避免一次异常退出导致整车打符彻底不可用
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
# 由 autostart.sh 自动生成，请勿手工修改
cd "$REPO_DIR" || exit 1
while true; do
  echo "===== \$(date '+%F %T') 启动 $PROGRAM ====="
  "$BIN" "$CONFIG"
  code=\$?
  echo "===== \$(date '+%F %T') $PROGRAM 退出 code=\$code ====="
  [ "\$code" -eq 0 ] && break      # 正常退出（例如按 q）就不再拉起
  echo "2 秒后自动重启..."
  sleep 2
done
EOF
chmod +x "$RUNNER"

if ! bash -n "$RUNNER"; then
  echo "[autostart] 生成的守护脚本语法有误: $RUNNER"
  exit 1
fi

SCREEN_CMD=(screen -L -Logfile "logs/$STAMP.screenlog" -S "$SCREEN_NAME" -d -m "$RUNNER")

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "[autostart] DRY_RUN=1，未真正启动。将要执行:"
  printf '  %q' "${SCREEN_CMD[@]}"
  echo
  echo "  守护脚本内容 ($RUNNER):"
  sed 's/^/    /' "$RUNNER"
  exit 0
fi

if screen -ls 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "[autostart] 已有名为 ${SCREEN_NAME} 的 screen 会话在运行，先退出它再启动："
  echo "           screen -S ${SCREEN_NAME} -X quit"
  exit 1
fi

"${SCREEN_CMD[@]}" || { echo "[autostart] screen 启动失败"; exit 1; }

echo
echo "[autostart] 已后台启动。"
echo "           进入会话: screen -r ${SCREEN_NAME}"
echo "           看日志  : tail -f logs/$STAMP.screenlog"
echo "           停止    : screen -S ${SCREEN_NAME} -X quit"
