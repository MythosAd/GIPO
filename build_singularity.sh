#!/bin/bash

# ==============================================================================
#  脚本说明 (无变化)
# ==============================================================================
# ... (和之前一样) ...
# ==============================================================================

# --- 配置段 ---
# 如果任何命令失败，脚本将立即退出
set -e

# --- 关键修复: 初始化 Module 环境 ---
# 不同的HPC系统，module的初始化脚本位置可能不同。
# /etc/profile.d/modules.sh 是最常见的位置。
if [ -f /etc/profile.d/modules.sh ]; then
   source /etc/profile.d/modules.sh
fi
# --- 修复结束 ---

# --- 镜像和文件配置 (无变化) ---
DOCKER_IMAGE_NAME="verlai/verl"
DOCKER_IMAGE_TAG="app-verl0.4-vllm0.8.5-mcore0.13.0-preview"
DOCKER_MIRROR_URL="https://docker.xuanyuan.me/"

# --- 输出文件配置 (无变化) ---
OUTPUT_SIF_FILE="my_project_image.sif"
TAR_ARCHIVE_FILE="my_project_image.tar"

# --- 目录配置 (无变化) ---
DOCKER_DATA_DIR="$(pwd)/docker_data_root"
SINGULARITY_TMP_DIR="$(pwd)/singularity_tmp"
SINGULARITY_CACHE_DIR="$(pwd)/singularity_cache"


# ==============================================================================
#  脚本主逻辑开始 (无变化)
# ==============================================================================
# ... (脚本的其余所有部分都和之前完全一样) ...
# ... (从 echo "===" 开始，到脚本结束) ...
# ==============================================================================

echo "============================================================"
echo "===       开始构建 Singularity SIF 文件                  ==="
echo "============================================================"
echo
echo "--> 脚本配置:"
echo "    Docker 镜像: ${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo "    镜像加速器 : ${DOCKER_MIRROR_URL}"
echo "    输出 SIF 文件: ${OUTPUT_SIF_FILE}"
echo

# --- 准备工作: 检查并创建目录 ---
echo "--> 正在准备目录..."
mkdir -p "$DOCKER_DATA_DIR"
mkdir -p "$SINGULARITY_TMP_DIR"
mkdir -p "$SINGULARITY_CACHE_DIR"
echo "    目录准备就绪。"
echo

# --- 阶段一: Docker 操作 (拉取并保存) ---
echo "============================================================"
echo "===       阶段一: Docker 操作                            ==="
echo "============================================================"

# 1. 加载 Docker 模块
echo "--> 正在加载 Docker 模块 (dock/dock6.9)..."
module unload singularity/3.10 || true # 忽略卸载失败的错误
module load dock/dock6.9
echo "    Docker 模块加载成功。"
echo

# 2. 为 Docker 配置镜像和数据根目录
echo "--> 正在为 Docker 配置用户级 daemon.json..."
mkdir -p ~/.config/docker
cat <<EOF > ~/.config/docker/daemon.json
{
  "registry-mirrors": ["${DOCKER_MIRROR_URL}"],
  "data-root": "${DOCKER_DATA_DIR}"
}
EOF
echo "    配置文件已写入 ~/.config/docker/daemon.json"
echo

# 3. 重启 Docker 服务以应用配置
echo "--> 正在重载 Docker 模块以应用新配置..."
module unload dock/dock6.9
module load dock/dock6.9
echo "    Docker 已使用新配置重启。"
echo

# 4. 拉取 Docker 镜像
echo "--> 正在拉取 Docker 镜像 (此过程可能非常耗时)..."
docker pull "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo "    镜像拉取成功！"
echo

# 5. 保存 Docker 镜像为 .tar 文件
echo "--> 正在将镜像保存为 ${TAR_ARCHIVE_FILE}..."
docker save -o "${TAR_ARCHIVE_FILE}" "${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}"
echo "    镜像保存成功！"
echo


# --- 阶段二: Singularity 操作 (构建 SIF) ---
echo "============================================================"
echo "===       阶段二: Singularity 操作                       ==="
echo "============================================================"

# 1. 加载 Singularity 模块
echo "--> 正在加载 Singularity 模块 (singularity/3.10)..."
module unload dock/dock6.9
module load singularity/3.10
echo "    Singularity 模块加载成功。"
echo

# 2. 从 .tar 文件构建 SIF
echo "--> 正在从 ${TAR_ARCHIVE_FILE} 构建 SIF 文件 (此过程可能耗时)..."
env \
  SINGULARITY_TMPDIR="${SINGULARITY_TMP_DIR}" \
  SINGULARITY_CACHEDIR="${SINGULARITY_CACHE_DIR}" \
  singularity build "${OUTPUT_SIF_FILE}" "docker-archive://${TAR_ARCHIVE_FILE}"
echo "    SIF 文件构建成功！"
echo


# --- 阶段三: 清理工作 ---
echo "============================================================"
echo "===       阶段三: 清理中间文件                         ==="
echo "============================================================"

echo "--> 正在删除 Docker 数据目录: ${DOCKER_DATA_DIR}"
rm -rf "${DOCKER_DATA_DIR}"
echo "    Docker 数据目录已删除。"
echo

echo "--> 正在删除 .tar 归档文件: ${TAR_ARCHIVE_FILE}"
rm -f "${TAR_ARCHIVE_FILE}"
echo "    .tar 归档文件已删除。"
echo


echo "============================================================"
echo "===       任务成功完成!                                  ==="
echo "===                                                      ==="
echo "===       最终产物: ${OUTPUT_SIF_FILE}                  ==="
echo "============================================================"