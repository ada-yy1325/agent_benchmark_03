#!/bin/bash
# ============================================================================
# PD Mooncake 一键打包脚本 - 防丢失
# 将所有 P/D 相关状态打包到共享盘持久化目录
# 用法: bash pack_pd_snapshot.sh [restore]
#   - 无参数: 创建快照
#   - restore: 从快照恢复到运行环境中
# ============================================================================
set -euo pipefail

SNAPSHOT_DIR="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/pd_snapshot"
PATCH_DIR_SRC="/inspire/sj-ssd3/project/project-public/s26068/agent_benchmark_test/perf_opt/scripts/pd_mooncake_patches"
CONNECTOR="/opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
ENGINE_SO="/opt/mamba/lib/python3.13/site-packages/mooncake/libtransfer_engine.so"
MC_COMMON_SO="/opt/mamba/lib/python3.13/site-packages/mooncake/libmooncake_common.so"
MC_YAML="/opt/mamba/lib/libyaml-cpp.so.0.8"
MC_ASIO="/opt/mamba/lib/python3.13/site-packages/mooncake/libasio.so"
MOONCAKE_DIR="/opt/mamba/lib/python3.13/site-packages/mooncake"
PRISTINE_CONNECTOR="/tmp/mc_0230.py"

if [ "${1:-}" = "restore" ]; then
    echo "[restore] 从快照恢复所有 P/D 组件..."
    
    # 1. 恢复引擎
    if [ -f "$SNAPSHOT_DIR/libtransfer_engine.so" ]; then
        cp "$SNAPSHOT_DIR/libtransfer_engine.so" "$ENGINE_SO"
        echo "  ✓ engine.so restored"
    fi
    
    # 2. 恢复 connector
    if [ -f "$SNAPSHOT_DIR/mooncake_connector_patched.py" ]; then
        cp "$SNAPSHOT_DIR/mooncake_connector_patched.py" "$CONNECTOR"
        echo "  ✓ mooncake_connector.py restored"
    fi
    
    # 3. 重新应用所有补丁（确保引擎+connector都是最新状态）
    echo "  re-applying patches..."
    python3 "$PATCH_DIR_SRC/patch_final_conn.py"
    python3 "$PATCH_DIR_SRC/patch_datalen.py"
    echo "  ✓ patches re-applied"
    
    echo "[restore] 完成。现在可以运行 regenerate_pd.py 重新生成启动脚本。"
    exit 0
fi

# ==================== SAVE ====================
echo "[pack] 创建 P/D 快照: $SNAPSHOT_DIR"
mkdir -p "$SNAPSHOT_DIR"

# 1. 备份引擎 .so 文件
echo "[pack] 1/8 引擎库..."
if [ -f "$ENGINE_SO" ]; then
    cp -v "$ENGINE_SO" "$SNAPSHOT_DIR/"
fi
if [ -f "$MC_COMMON_SO" ]; then
    cp -v "$MC_COMMON_SO" "$SNAPSHOT_DIR/"
fi
if [ -f "$MC_YAML" ]; then
    cp -v "$MC_YAML" "$SNAPSHOT_DIR/libyaml-cpp.so.0.8"
fi

# 2. 备份当前 connector（已打补丁）
echo "[pack] 2/8 当前 connector..."
cp -v "$CONNECTOR" "$SNAPSHOT_DIR/mooncake_connector_patched.py"

# 3. 备份原始 pristine connector
echo "[pack] 3/8 pristine connector..."
if [ -f "$PRISTINE_CONNECTOR" ]; then
    cp -v "$PRISTINE_CONNECTOR" "$SNAPSHOT_DIR/"
fi

# 4. 备份所有补丁脚本
echo "[pack] 4/8 补丁脚本..."
cp -rv "$PATCH_DIR_SRC"/*.py "$SNAPSHOT_DIR/" 2>/dev/null || true
cp -v "$PATCH_DIR_SRC/README.md" "$SNAPSHOT_DIR/"

# 5. 备份 P/D 启动脚本和参数
echo "[pack] 5/8 启动参数..."
if [ -f /tmp/pd_test/pd_prefill_args.json ]; then
    cp -v /tmp/pd_test/pd_prefill_args.json "$SNAPSHOT_DIR/"
fi
if [ -f /tmp/pd_test/pd_decode_args.json ]; then
    cp -v /tmp/pd_test/pd_decode_args.json "$SNAPSHOT_DIR/"
fi
if [ -f /tmp/pd_test/_pd_prefill_launch.sh ]; then
    cp -v /tmp/pd_test/_pd_prefill_launch.sh "$SNAPSHOT_DIR/"
fi
if [ -f /tmp/pd_test/_pd_decode_launch.sh ]; then
    cp -v /tmp/pd_test/_pd_decode_launch.sh "$SNAPSHOT_DIR/"
fi

# 6. 备份 vLLM 源码（如有改动）
echo "[pack] 6/8 vLLM 相关源码..."
find /opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/ \
    -name "*.py" -exec cp --parents {} "$SNAPSHOT_DIR/" \; 2>/dev/null || true

# 7. 备份环境信息
echo "[pack] 7/8 环境信息..."
python3 -c "
import sys, torch, vllm, json
info = {
    'python': sys.version,
    'pytorch': torch.__version__,
    'vllm': vllm.__version__,
    'torch_npu': getattr(torch, 'npu', None) is not None,
}
json.dump(info, open('$SNAPSHOT_DIR/env_info.json', 'w'), indent=2)
print(json.dumps(info, indent=2))
"

# 8. 备份 pd_test 工作目录
echo "[pack] 8/8 pd_test 工作目录..."
if [ -d /tmp/pd_test ]; then
    cp -rv /tmp/pd_test/*.py /tmp/pd_test/*.sh /tmp/pd_test/*.json "$SNAPSHOT_DIR/" 2>/dev/null || true
fi

# 生成恢复脚本
echo "[pack] 生成 restore.sh..."
cat > "$SNAPSHOT_DIR/restore.sh" << 'RESTORE_EOF'
#!/bin/bash
# ============================================================================
# 一键恢复: 从快照恢复所有 P/D 组件到运行环境中
# 用法: bash restore.sh
# ============================================================================
set -euo pipefail
SNAP="$(cd "$(dirname "$0")" && pwd)"

echo "[restore] 恢复 P/D Mooncake 组件..."

# 引擎
if [ -f "$SNAP/libtransfer_engine.so" ]; then
    cp "$SNAP/libtransfer_engine.so" /opt/mamba/lib/python3.13/site-packages/mooncake/
fi
if [ -f "$SNAP/libmooncake_common.so" ]; then
    cp "$SNAP/libmooncake_common.so" /opt/mamba/lib/python3.13/site-packages/mooncake/
fi
if [ -f "$SNAP/libyaml-cpp.so.0.8" ]; then
    cp "$SNAP/libyaml-cpp.so.0.8" /opt/mamba/lib/
fi

# 恢复 pristine connector 并重新打补丁
if [ -f "$SNAP/mc_0230.py" ]; then
    cp "$SNAP/mc_0230.py" /opt/mamba/lib/python3.13/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
    python3 "$SNAP/patch_final_conn.py"
    python3 "$SNAP/patch_datalen.py"
    echo "[restore] connector patched"
fi

echo "[restore] 完成。运行 regenerate_pd.py 重新生成启动脚本。"
RESTORE_EOF
chmod +x "$SNAPSHOT_DIR/restore.sh"

# 清单
echo ""
echo "============================================"
echo "  P/D 快照已保存到: $SNAPSHOT_DIR"
echo "  一键恢复: bash $SNAPSHOT_DIR/restore.sh"
echo "============================================"
ls -lh "$SNAPSHOT_DIR/" | head -30