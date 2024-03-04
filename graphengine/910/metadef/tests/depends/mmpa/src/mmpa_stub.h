/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef METADEF_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_
#define METADEF_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_

#include <cstdint>
#include <memory>
#include "mmpa/mmpa_api.h"

namespace ge {
class MmpaStubApi {
 public:
  virtual ~MmpaStubApi() = default;

  virtual void *DlOpen(const char *file_name, int32_t mode) {
    return dlopen(file_name, mode);
  }

  virtual void *DlSym(void *handle, const char *func_name) {
    return dlsym(handle, func_name);
  }

  virtual CHAR *Dlerror() {
    return dlerror();
  }

  virtual int32_t DlClose(void *handle) {
    return dlclose(handle);
  }

  virtual int32_t RealPath(const CHAR *path, CHAR *realPath, INT32 realPathLen) {
    INT32 ret = EN_OK;
    char *ptr = realpath(path, realPath);
    if (ptr == nullptr) {
      ret = EN_ERROR;
    }
    return ret;
  };

  virtual INT32 WaitPid(mmProcess pid, INT32 *status, INT32 options) {
    if ((options != MMPA_ZERO) && (options != M_WAIT_NOHANG) && (options != M_WAIT_UNTRACED)) {
    return EN_INVALID_PARAM;
    }

    INT32 ret = waitpid(pid, status, options);
    if (ret == EN_ERROR) {
      return EN_ERROR;
    }
    if ((ret > MMPA_ZERO) && (ret == pid)) {
      if (status != NULL) {
        if (WIFEXITED(*status)) {
          *status = WEXITSTATUS(*status);
        }
        if(WIFSIGNALED(*status)) {
          *status = WTERMSIG(*status);
        }
      }
      return EN_ERR;
    }
    return EN_OK;
  }

  virtual INT32 mmGetTimeOfDay(mmTimeval *timeVal, mmTimezone *timeZone)
  {
    if (timeVal == nullptr) {
      return EN_ERR;
    }
    return gettimeofday(reinterpret_cast<timeval *>(timeVal), nullptr);
  }

  virtual INT32 mmGetSystemTime(mmSystemTime_t *sysTime) {
    if (sysTime == nullptr) {
      return EN_ERR;
    }
    time_t cur_time;
    time(&cur_time);
    tm *now_time = localtime(&cur_time);
    sysTime->wYear = 1900 + now_time->tm_year;
    sysTime->wMonth = 1+ now_time->tm_mon;
    sysTime->wDay = now_time->tm_mday;
    sysTime->wHour = now_time->tm_hour;
    sysTime->wMinute = now_time->tm_min;
    sysTime->wSecond = now_time->tm_sec;
    return EN_OK;
  }
};

class MmpaStub {
 public:
  static MmpaStub& GetInstance() {
    static MmpaStub instance;
    return instance;
  }

  void SetImpl(const std::shared_ptr<MmpaStubApi> &impl) {
    impl_ = impl;
  }

  MmpaStubApi* GetImpl() {
    return impl_.get();
  }

  void Reset() {
    impl_ = std::make_shared<MmpaStubApi>();
  }

 private:
  MmpaStub(): impl_(std::make_shared<MmpaStubApi>()) {
  }

  std::shared_ptr<MmpaStubApi> impl_;
};

}  // namespace ge

#endif  // AIR_TESTS_DEPENDS_MMPA_SRC_MMAP_STUB_H_
