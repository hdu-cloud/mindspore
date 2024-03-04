#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

set -e
set -u

# The version of opensdk is tagged as ${YearMonthDay}. If the user-specified 
# ${YearMonthDay} is earlier than 20230820, please use the original link below:
# REPO_HOST="https://ascend-repo.obs.myhuaweicloud.com"
# REPO_TYPE="CANN_daily_y2b"

REPO_HOST="https://ascend-cann.obs.myhuaweicloud.com"
REPO_TYPE="CANN"
FILE_NAME="ai_cann_x86.tar.gz"
HTTP_OK="200"

# Directly download the newest opensdk package from hdfs server.
HDFS_HOST="https://hdfs-ngx0.turing-ci.hisilicon.com"

usage() {
  echo "Usage:"
  echo "    bash update_newest_packages.sh [-k] [-d <path>] [-t <YearMonthDay>]"
  echo "Description:"
  echo "    -k, Allow insecure server connections when using SSL."
  echo "    -d <dir_name>, Extract files into dir_name."
  echo "    -t <YearMonthDay>, Specify the package version. (e.g., -t 20230821)"
  echo "    -Y Download opensdk from HDFS server directly. (Yellow Zone only)"
}

checkopts(){
  PREFIX=""
  NEWEST_DATE=""
  INSTALL_DIR=""
  CA_OPTION=""
  ENABLE_HDFS="off"
  while getopts 'd:t:kY' opt; do
    case "${opt}" in
      k)
        CA_OPTION="-k"
        ;;
      d)
        PREFIX=$(realpath ${OPTARG})
        ;;
      t)
        NEWEST_DATE=${OPTARG}
        ;;
      Y)
        ENABLE_HDFS="on"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}

build_download_url() {
  local day=$1
  local url="${REPO_HOST}/${REPO_TYPE}/${day}_newest/${FILE_NAME}"
  echo ${url}
}

build_install_dir() {
  local base_dir=$1
  local pkg_name="${2}_newest"
  local path="${base_dir}/${pkg_name}"
  echo ${path}
}

verify_url() {
  local url=$1
  local status=0
  set +e
  status=$(curl -k -I -m 5 -o /dev/null -s -w %{http_code} ${url})
  set -e
  echo ${status}
}

probe_newest_package() {
  local url=""
  local status=0
  local start=$(date +%Y%m%d)
  local end=$(date -d "-28 day ${start}" +%Y%m%d)
  echo "Start to probe the newest package from ${start} to ${end}"
  while [ ${start} -ge ${end} ]; do
    url=$(build_download_url ${start})
    if [[ $(verify_url ${url}) -eq ${HTTP_OK} ]]; then
      # find the latest package and update the global variable 
      NEWEST_DATE=${start}
      echo "The latest package version found: ${NEWEST_DATE}_newest/${FILE_NAME}"
      break
    fi
    start=$(date -d "-1 day ${start}" +%Y%m%d)
    url=""
    # set request interval to avoid potential Access Denied
    sleep 0.2
  done

  if [[ -z ${url} ]]; then 
    echo "Failed to find the newest package in the last 28 days. Please Check..."
    exit 1
  fi
}

extract_and_install_package() {
  local install_path=$1
  local tar_name=$2
  echo "Extracting..."
  cd ${install_path}
  # extract tar.gz first if downloaded from Blue Zone
  if [[ "X${ENABLE_HDFS}" = "Xoff" ]]; then
    tar -xf ${tar_name}
    cp ai_cann_x86/*opensdk*.run ./
    rm -rf ai_cann_x86
  fi
  # the newest opensdk is a run package
  local pkg=$(ls ./ | grep opensdk)
  # run the package
  echo ${pkg}
  chmod u+x ${pkg}
  if test -x ${pkg}; then
    ./${pkg} --noexec --extract=./opensdk
  fi
  # create symbolic link in Prefix/ 
  cd ../
  rm -rf latest
  ln -s ${install_path} latest
}

search_hdfs_file_list() {
  # YearMonthDay
  local version=$1
  local year_month=$(echo ${version} | cut -c 1-6)
  local regex="^${version}_[^_]*_newest$"
  local key="pathSuffix"
  # Step1: download file list (opensdk packages are grouped by YearMonth)
  local info_path="webhdfs/v1/compilepackage/CI_Version/cann/br_hisi_trunk_ai/${year_month}"
  local op_user="op=LISTSTATUS&user.name=balong"
  local file_list_url="${HDFS_HOST}/${info_path}?${op_user}"
  local json=$(curl -s ${CA_OPTION} ${file_list_url})
  # Step2: find the package version from file list
  version=$(echo ${json} | awk -F"[,:}]" '{for(i=1;i<=NF;i++){if($i~/'${key}'\042/){print $(i+1)}}}' | \
            tr -d '"' | grep ${regex} | tail -1)
  file_list_url="${HDFS_HOST}/${info_path}/${version}?${op_user}"
  json=$(curl -s ${CA_OPTION} ${file_list_url})
  local package_name=$(echo ${json} | awk -F"[,:}]" '{for(i=1;i<=NF;i++){if($i~/'${key}'\042/){print $(i+1)}}}' | \
                       tr -d '"' | grep "^CANN-opensdk" | grep "x86_64")
  local download_url="${HDFS_HOST}/${info_path}/${version}/${package_name}?op=OPEN&user.name=balong"
  echo ${download_url}
}

checkopts "$@"

if [[ ${NEWEST_DATE} =~ ^20 ]]; then
  # user-specified package version
  echo "Download package with user-specified version ${NEWEST_DATE}"
  DOWNLOAD_URL=$(build_download_url ${NEWEST_DATE})
  if [[ $(verify_url ${DOWNLOAD_URL}) -ne ${HTTP_OK} ]]; then
    echo "Invalid package version: ${NEWEST_DATE}_newest. Please check..."
    exit 1
  fi
else
  # automatically probe package version
  probe_newest_package
  DOWNLOAD_URL=$(build_download_url ${NEWEST_DATE})
fi

if [[ "X${ENABLE_HDFS}" = "Xon" ]]; then
  # only enabled in Yellow Zone
  if [[ $(verify_url ${HDFS_HOST}) -ne ${HTTP_OK} ]]; then
    echo "The HDFS_HOST(${HDFS_HOST}) is unavailable. Please check..."
    exit 1
  fi
  DOWNLOAD_URL=$(search_hdfs_file_list ${NEWEST_DATE})
  FILE_NAME="CANN-opensdk-x86_64.run"
fi

if [[ -z ${PREFIX} ]]; then 
  echo "Please specified the path to download and install the package."
  usage
  exit 1
fi

INSTALL_DIR=$(build_install_dir ${PREFIX} ${NEWEST_DATE})

echo "-- Download url: ${DOWNLOAD_URL}"
echo "-- Prefix: ${PREFIX}"
echo "-- Local install path: ${INSTALL_DIR}"
if [ -d ${INSTALL_DIR} ]; then
    echo "The newest package already exists, no need to update."
    echo "Newest package path: ${INSTALL_DIR}."
    exit 0
fi
echo "Download the newest package from ${DOWNLOAD_URL} to ${INSTALL_DIR}/${FILE_NAME}..."
mkdir -p ${INSTALL_DIR}
curl ${CA_OPTION} -o ${INSTALL_DIR}/${FILE_NAME} ${DOWNLOAD_URL}

extract_and_install_package ${INSTALL_DIR} ${FILE_NAME}

echo "Updated successfully!"
echo "When you build in metadef: set cmake option -DASCEND_OPENSDK_DIR=${PREFIX}/latest/opensdk/opensdk/"
echo "When you build in air: set env variable ASCEND_CUSTOM_PATH=${PREFIX}/latest"