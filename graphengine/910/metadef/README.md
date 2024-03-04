# metadef

## 介绍

昇腾元数据定义

## 许可证

[Apache License 2.0](LICENSE)

## clion配置
step1: download opensdk:
http://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN_daily_y2b/20221020_newest/ai_cann_x86.tar.gz

step2: unzip opendsdk
cp ai_cann_x86.tar.gz /home/pkg
tar -xvf ai_cann_x86.tar.gz
cd  ai_cann_x86
tar -xvf CANN-opensdk*.tar.gz

step3: Clion cmake config
-DENABLE_OPEN_SRC=True -DBUILD_WITHOUT_AIR=True -DENABLE_GITEE=True -DENABLE_METADEF_UT=True -DENABLE_METADEF_ST=True -DASCEND_OPENSDK_DIR=/home/pkg/ai_cann_x86/opensdk/opensdk -DCMAKE_INSTALL_PREFIX=/home/pkg/ai_cann_x86

You can replace /home/pkg with other path
