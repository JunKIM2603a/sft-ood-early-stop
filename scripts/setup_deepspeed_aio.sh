#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOCAL_ROOT="${ROOT_DIR}/artifacts/local/libaio"
PKG_DIR="${LOCAL_ROOT}/packages"
EXTRACT_DIR="${LOCAL_ROOT}/root"
ENV_FILE="${ROOT_DIR}/artifacts/env/deepspeed_aio.env"

mkdir -p "${PKG_DIR}" "${EXTRACT_DIR}" "$(dirname "${ENV_FILE}")"

check_aio() {
  python - <<'PY'
from deepspeed.ops.op_builder import AsyncIOBuilder
ok = bool(AsyncIOBuilder().is_compatible())
print(f"DeepSpeed async_io compatible: {ok}")
raise SystemExit(0 if ok else 1)
PY
}

write_env_file() {
  local include_dir="$1"
  local lib_dir="$2"

  cat > "${ENV_FILE}" <<EOF_ENV
export CFLAGS="-I${include_dir} ${CFLAGS:-}"
export LDFLAGS="-L${lib_dir} ${LDFLAGS:-}"
export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
EOF_ENV

  echo "Wrote ${ENV_FILE}"
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
}

package_candidate_version() {
  local package="$1"
  local version

  version="$(
    LC_ALL=C apt-cache policy "${package}"       | awk '/Candidate:/ {print $2; exit}'
  )"

  if [[ -z "${version}" || "${version}" == "(none)" ]]; then
    version="$(
      LC_ALL=C apt-cache show "${package}" 2>/dev/null         | awk '/^Version:/ {print $2; exit}'
    )"
  fi

  if [[ -z "${version}" ]]; then
    echo "ERROR: could not resolve a version for ${package} from apt metadata." >&2
    return 1
  fi

  printf '%s\n' "${version}"
}

runtime_package_name() {
  local package="$1"
  local runtime

  runtime="$(
    LC_ALL=C apt-cache depends "${package}"       | awk '/Depends: libaio/ {gsub(/[<>]/, "", $2); print $2; exit}'
  )"

  if [[ -z "${runtime}" ]]; then
    if LC_ALL=C apt-cache show libaio1 >/dev/null 2>&1; then
      runtime="libaio1"
    elif LC_ALL=C apt-cache show libaio1t64 >/dev/null 2>&1; then
      runtime="libaio1t64"
    fi
  fi

  printf '%s\n' "${runtime}"
}

echo "=== DeepSpeed async_io / libaio setup ==="

if check_aio; then
  echo "PASS: async_io is already compatible."
  exit 0
fi

if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  echo "Passwordless sudo is available; installing system libaio-dev."
  sudo apt-get update
  sudo apt-get install -y libaio-dev

  if check_aio; then
    echo "PASS: async_io is compatible after system libaio-dev install."
    exit 0
  fi

  echo "ERROR: libaio-dev installed, but DeepSpeed async_io remains incompatible."
  echo "Run: ds_report"
  exit 1
fi

echo "No passwordless sudo. Using user-local Debian package extraction."

for cmd in apt-get apt-cache dpkg-deb dpkg; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: ${cmd} is required for the non-root libaio setup."
    echo "Administrator fallback: sudo apt-get install -y libaio-dev"
    exit 1
  fi
done

rm -rf "${PKG_DIR}" "${EXTRACT_DIR}"
mkdir -p "${PKG_DIR}" "${EXTRACT_DIR}"

pushd "${PKG_DIR}" >/dev/null

download_ubuntu_pool_package() {
  local filename="$1"
  local mirrors=(
    "https://kr.archive.ubuntu.com/ubuntu"
    "https://pl.archive.ubuntu.com/ubuntu"
    "https://us.archive.ubuntu.com/ubuntu"
    "https://archive.ubuntu.com/ubuntu"
    "https://old-releases.ubuntu.com/ubuntu"
  )

  for mirror in "${mirrors[@]}"; do
    local url="${mirror}/pool/main/liba/libaio/${filename}"
    echo "Trying Ubuntu mirror: ${url}"

    if command -v curl >/dev/null 2>&1; then
      if curl -fL --retry 2 --retry-delay 1 -o "${filename}" "${url}"; then
        echo "Downloaded ${filename} from ${mirror}"
        return 0
      fi
    elif command -v wget >/dev/null 2>&1; then
      if wget -O "${filename}" "${url}"; then
        echo "Downloaded ${filename} from ${mirror}"
        return 0
      fi
    else
      echo "ERROR: curl or wget is required for Ubuntu mirror fallback."
      return 1
    fi

    rm -f "${filename}"
  done

  echo "ERROR: failed to download ${filename} from all Ubuntu mirrors."
  return 1
}

if ! apt-get download libaio-dev; then
  echo "WARN: configured APT mirror failed for libaio-dev; trying Ubuntu archive mirrors."
  DEV_VERSION="$(package_candidate_version libaio-dev)"
  ARCH="$(dpkg --print-architecture)"
  DEV_FILE="libaio-dev_${DEV_VERSION}_${ARCH}.deb"
  echo "Resolved libaio-dev version: ${DEV_VERSION}"
  download_ubuntu_pool_package "${DEV_FILE}"
fi

RUNTIME_PKG="$(runtime_package_name libaio-dev)"

if [[ -n "${RUNTIME_PKG}" ]]; then
  echo "Detected libaio runtime dependency: ${RUNTIME_PKG}"
  if ! apt-get download "${RUNTIME_PKG}"; then
    echo "WARN: configured APT mirror failed for ${RUNTIME_PKG}; trying Ubuntu archive mirrors."
    RUNTIME_VERSION="$(package_candidate_version "${RUNTIME_PKG}")"
    ARCH="$(dpkg --print-architecture)"
    RUNTIME_FILE="${RUNTIME_PKG}_${RUNTIME_VERSION}_${ARCH}.deb"
    echo "Resolved ${RUNTIME_PKG} version: ${RUNTIME_VERSION}"
    download_ubuntu_pool_package "${RUNTIME_FILE}"
  fi
else
  echo "ERROR: could not resolve libaio runtime dependency."
  exit 1
fi

shopt -s nullglob
DEBS=( *.deb )
if [[ "${#DEBS[@]}" -eq 0 ]]; then
  echo "ERROR: no libaio .deb files were downloaded."
  exit 1
fi

for deb in "${DEBS[@]}"; do
  echo "Extracting ${deb}"
  dpkg-deb -x "${deb}" "${EXTRACT_DIR}"
done

popd >/dev/null

INCLUDE_DIR="${EXTRACT_DIR}/usr/include"
if [[ ! -f "${INCLUDE_DIR}/libaio.h" ]]; then
  echo "ERROR: local extraction did not produce libaio.h"
  exit 1
fi

LIBAIO_SO="$(
  find "${EXTRACT_DIR}/usr/lib" "${EXTRACT_DIR}/lib"     \( -type f -o -type l \) 2>/dev/null     | grep -E '/libaio\.so($|\.)'     | head -n 1 || true
)"

if [[ -z "${LIBAIO_SO}" ]]; then
  echo "ERROR: local extraction did not produce libaio.so*"
  exit 1
fi

LIB_DIR="$(dirname "${LIBAIO_SO}")"
write_env_file "${INCLUDE_DIR}" "${LIB_DIR}"

if ! check_aio; then
  echo "ERROR: local libaio is present but DeepSpeed still reports async_io incompatible."
  echo "Environment file: ${ENV_FILE}"
  echo "Run after sourcing it: ds_report"
  exit 1
fi

echo "PASS: user-local libaio enables DeepSpeed async_io."
echo "Future P2 scripts source: ${ENV_FILE}"
