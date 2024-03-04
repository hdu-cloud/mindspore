
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_def_registry.h"

namespace ops {

namespace {

class OpDefAICoreUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefAICoreUT, AICoreTest) {
  OpAICoreDef aicoreDef;
  OpAICoreConfig config;
  config.DynamicCompileStaticFlag(true)
      .DynamicFormatFlag(true)
      .DynamicRankSupportFlag(true)
      .DynamicShapeSupportFlag(true)
      .NeedCheckSupportFlag(true)
      .PrecisionReduceFlag(true);
  std::map<ge::AscendString, ge::AscendString> cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "true");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");
  config.DynamicCompileStaticFlag(false)
      .DynamicFormatFlag(false)
      .DynamicRankSupportFlag(false)
      .DynamicShapeSupportFlag(false)
      .NeedCheckSupportFlag(false)
      .PrecisionReduceFlag(false);
  cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "false");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "false");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "false");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "false");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "false");
  aicoreDef.AddConfig("ascend310p", config);
  aicoreDef.AddConfig("ascend910", config);
  aicoreDef.AddConfig("ascend310p", config);
  std::map<ge::AscendString, OpAICoreConfig> aicfgs = aicoreDef.GetAICoreConfigs();
  EXPECT_TRUE(aicfgs.find("ascend310p") != aicfgs.end());
  EXPECT_EQ(aicfgs.size(), 2);
  aicoreDef.AddConfig("ascend310p");
  aicfgs = aicoreDef.GetAICoreConfigs();
  config = aicfgs["ascend310p"];
  cfgs = config.GetCfgInfo();
  EXPECT_EQ(cfgs["dynamicCompileStatic.flag"], "true");
  EXPECT_EQ(cfgs["dynamicFormat.flag"], "true");
  EXPECT_EQ(cfgs["dynamicRankSupport.flag"], "true");
  EXPECT_EQ(cfgs["dynamicShapeSupport.flag"], "true");
  EXPECT_EQ(cfgs["needCheckSupport.flag"], "false");
  EXPECT_EQ(cfgs["precision_reduce.flag"], "true");
}

}  // namespace
}  // namespace ops
