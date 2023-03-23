#!/bin/bash
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
BUILD_RELATIVE_PATH="build"
BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"

# print usage message
usage()
{
  echo "Usage:"
  echo "sh build.sh [-j[n]] [-h] [-v] [-s] [-b] [-t] [-u] [-c] [-S on|off]"
  echo ""
  echo "Options:"
  echo "    -h Print usage"
  echo "    -u Only compile ut, not execute"
  echo "    -s Build st"
  echo "    -j[n] Set the number of threads used for building Metadef, default is 8"
  echo "    -t Build and execute ut"
  echo "    -b Run Benchmark test"
  echo "    -c Build ut with coverage tag"
  echo "    -v Display build command"
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
  echo "to be continued ..."
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# parse and set options
checkopts()
{
  VERBOSE=""
  THREAD_NUM=8
  # ENABLE_METADEF_UT_ONLY_COMPILE="off"
  ENABLE_METADEF_UT="off"
  ENABLE_METADEF_ST="off"
  ENABLE_METADEF_COV="off"
  ENABLE_BENCHMARK="off"
  GE_ONLY="on"
  ENABLE_GITEE="off"
  # Process the options
  while getopts 'ustcbhj:vS:' opt
  do
    OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')
    case "${opt}" in
      u)
        ENABLE_METADEF_UT="on"
        GE_ONLY="off"
        ;;
      s)
        ENABLE_METADEF_ST="on"
        ;;
      t)
        ENABLE_METADEF_UT="on"
        GE_ONLY="off"
        ;;
      c)
        ENABLE_METADEF_COV="on"
        GE_ONLY="off"
        ;;
      b)
        ENABLE_BENCHMARK="on"
        GE_ONLY="off"
        ;;
      h)
        usage
        exit 0
        ;;
      j)
        THREAD_NUM=$OPTARG
        ;;
      v)
        VERBOSE="VERBOSE=1"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      *)
        echo "Undefined option: ${opt}"
        usage
        exit 1
    esac
  done
}
checkopts "$@"

mk_dir() {
    local create_dir="$1"  # the target to make

    mkdir -pv "${create_dir}"
    echo "created ${create_dir}"
}

# Meatdef build start
echo "---------------- Metadef build start ----------------"

function get_metadef_lib_dir() {
  if [ ${D_LINK_PATH} ] ; then
    local arch=$(uname -m)
    if [[ "${arch}" == "x86_64" || "${arch}" == "aarch64" ]]; then
      echo ${D_LINK_PATH}/${arch}
    else
      echo "metadef lib dir config error !!"
      exit 1
    fi
  elif [ ${ASCEND_CUSTOM_PATH} ]; then
    echo ${ASCEND_CUSTOM_PATH}/compiler/lib64
  fi
}

function cmake_generate_make() {
  local build_path=$1;
  local cmake_args=$2;
  mk_dir "${build_path}"
  cd "${build_path}"
  echo "${cmake_args}"
  cmake ${cmake_args} ..
  if [ 0 -ne $? ]
  then
    echo "execute command: cmake ${CMAKE_ARGS} .. failed."
    exit 1
  fi
}

# create build path
build_metadef()
{
  echo "create build directory and build Metadef";

  if [[ "X$ENABLE_METADEF_UT" = "Xon" || "X$ENABLE_METADEF_COV" = "Xon" ]]; then
    BUILD_RELATIVE_PATH="build_gcov"
    CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=GCOV"
  fi

  if [[ "X$ENABLE_METADEF_UT" = "Xon" || "X$ENABLE_METADEF_COV" = "Xon" || "X$ENABLE_BENCHMARK" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS}"
  else
    local metadef_lib_dir=$(get_metadef_lib_dir);
    CMAKE_ARGS="${CMAKE_ARGS} -DMETADEF_LIB_DIR=${metadef_lib_dir}"
  fi

  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi

  BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
  CMAKE_ARGS="${CMAKE_ARGS} -DBUILD_PATH=$BUILD_PATH -DGE_ONLY=$GE_ONLY"
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_OPEN_SRC=True -DCMAKE_INSTALL_PREFIX=${OUTPUT_PATH}"
  cmake_generate_make "${BUILD_PATH}" "${CMAKE_ARGS}"

  if [[ "X$ENABLE_METADEF_UT" = "Xon" || "X$ENABLE_METADEF_COV" = "Xon" ]]; then
    make ut_graph ut_register ut_error_manager ut_exe_graph ${VERBOSE} -j${THREAD_NUM}
  elif [ "X$ENABLE_BENCHMARK" = "Xon" ]; then
    make exec_graph_benchmark ${VERBOSE} -j${THREAD_NUM}
  else
    make graph graph_static exe_graph register register_static error_manager error_manager_static ${VERBOSE} -j${THREAD_NUM} && make install
  fi
  if [ 0 -ne $? ]
  then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "Metadef build success!"
}

generate_inc_coverage() {
  echo "Generating inc coverage, please wait..."
  rm -rf ${BASEPATH}/diff
  mkdir -p ${BASEPATH}/cov/diff

  git diff --src-prefix=${BASEPATH}/ --dst-prefix=${BASEPATH}/ HEAD^ > ${BASEPATH}/cov/diff/inc_change_diff.txt
  addlcov --diff ${BASEPATH}/cov/coverage.info ${BASEPATH}/cov/diff/inc_change_diff.txt -o ${BASEPATH}/cov/diff/inc_coverage.info
  genhtml --prefix ${BASEPATH} -o ${BASEPATH}/cov/diff/html ${BASEPATH}/cov/diff/inc_coverage.info --legend -t CHG --no-branch-coverage --no-function-coverage
}

g++ -v
mk_dir ${OUTPUT_PATH}
build_metadef || { echo "Metadef build failed."; return; }
echo "---------------- Metadef build finished ----------------"
rm -f ${OUTPUT_PATH}/libgmock*.so
rm -f ${OUTPUT_PATH}/libgtest*.so
rm -f ${OUTPUT_PATH}/lib*_stub.so

chmod -R 750 ${OUTPUT_PATH}
find ${OUTPUT_PATH} -name "*.so*" -print0 | xargs -0 chmod 500

echo "---------------- Metadef output generated ----------------"

if [ "X$ENABLE_BENCHMARK" = "Xon" ]; then
  RUN_TEST_CASE= ${BUILD_PATH}/tests/benchmark/exec_graph_benchmark && ${RUN_TEST_CASE}
fi

if [[ "X$ENABLE_METADEF_UT" = "Xon" || "X$ENABLE_METADEF_COV" = "Xon" ]]; then
    cp ${BUILD_PATH}/tests/ut/graph/ut_graph ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/register/ut_register ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/error_manager/ut_error_manager ${OUTPUT_PATH}
    cp ${BUILD_PATH}/tests/ut/exe_graph/ut_exe_graph ${OUTPUT_PATH}

    RUN_TEST_CASE=${OUTPUT_PATH}/ut_graph && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_register && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_error_manager && ${RUN_TEST_CASE} &&
    RUN_TEST_CASE=${OUTPUT_PATH}/ut_exe_graph && ${RUN_TEST_CASE}
    if [[ "$?" -ne 0 ]]; then
        echo "!!! UT FAILED, PLEASE CHECK YOUR CHANGES !!!"
        echo -e "\033[31m${RUN_TEST_CASE}\033[0m"
        exit 1;
    fi
    echo "Generating coverage statistics, please wait..."
    cd ${BASEPATH}
    rm -rf ${BASEPATH}/cov
    mkdir ${BASEPATH}/cov
    lcov -c \
        -d ${BUILD_RELATIVE_PATH}/graph/CMakeFiles/graph.dir \
        -d ${BUILD_RELATIVE_PATH}/register/CMakeFiles/register.dir/ \
        -d ${BUILD_RELATIVE_PATH}/error_manager/CMakeFiles/error_manager.dir/ \
        -d ${BUILD_RELATIVE_PATH}/exe_graph/CMakeFiles/exe_graph.dir/ \
        -d ${BUILD_RELATIVE_PATH}/tests/ut/exe_graph/CMakeFiles/ut_exe_graph.dir \
        -o cov/tmp.info
    lcov -r cov/tmp.info '*/output/*' '*/${BUILD_RELATIVE_PATH}/opensrc/*' '*/${BUILD_RELATIVE_PATH}/proto/*' '*/third_party/*' '*/tests/*' '/usr/*' '*/ops/*' -o cov/coverage.info
    cd ${BASEPATH}/cov
    genhtml coverage.info
    if [[ "X$ENABLE_METADEF_COV" = "Xon" ]]; then
      generate_inc_coverage
    fi
fi

# generate output package in tar form, including ut/st libraries/executables for cann
generate_package()
{
  cd "${BASEPATH}"

  METADEF_LIB_PATH="lib"
  COMPILER_PATH="compiler/lib64"
  RUNTIME_PATH="runtime/lib64"
  COMMON_LIB=("libgraph.so" "libregister.so" "liberror_manager.so")

  rm -rf ${OUTPUT_PATH:?}/${COMPILER_PATH}/
  rm -rf ${OUTPUT_PATH:?}/${RUNTIME_PATH}/

  mk_dir "${OUTPUT_PATH}/${COMPILER_PATH}"
  mk_dir "${OUTPUT_PATH}/${RUNTIME_PATH}"

  find output/ -name metadef_lib.tar -exec rm {} \;

  cd "${OUTPUT_PATH}"

  for lib in "${COMMON_LIB[@]}";
  do
    find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
    find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "$lib" -exec cp -f {} ${OUTPUT_PATH}/${RUNTIME_PATH} \;
  done

  find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "libc_sec.so" -exec cp -f {} ${OUTPUT_PATH}/${COMPILER_PATH} \;
  find ${OUTPUT_PATH}/${METADEF_LIB_PATH} -maxdepth 1 -name "libc_sec.so" -exec cp -f {} ${OUTPUT_PATH}/${RUNTIME_PATH} \;

  tar -cf metadef_lib.tar compiler runtime
}

if [[ "X$ENABLE_METADEF_UT" = "Xoff" && "X$ENABLE_BENCHMARK" = "Xoff" ]]; then
  generate_package
fi
echo "---------------- Metadef package archive generated ----------------"
