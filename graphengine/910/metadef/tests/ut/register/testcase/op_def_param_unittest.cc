
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_def_registry.h"
#include "register/opdef/op_def_impl.h"

namespace ops {

namespace {

class OpDefParamUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefParamUT, ParamTest) {
  OpParamDef param("test");
  OpParamDef param2("test");
  OpParamDef param3("test3");
  EXPECT_EQ(param == param2, true);
  EXPECT_EQ(param == param3, false);
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_NCHW})
      .ValueDepend(Option::REQUIRED);
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED);
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL);
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED);
  EXPECT_EQ(desc.Input("x1").GetParamName(), "x1");
  EXPECT_EQ(desc.Input("x1").GetParamType(), Option::OPTIONAL);
  EXPECT_EQ(desc.Input("x1").GetDataTypes().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats()[0], ge::FORMAT_NCHW);
  EXPECT_EQ(desc.Input("x1").GetValueDepend(), "required");
  EXPECT_EQ(desc.Input("x2").GetValueDepend(), "optional");
  EXPECT_EQ(desc.GetInputs().size(), 2);
  EXPECT_EQ(desc.GetOutputs().size(), 1);
}

}  // namespace
}  // namespace ops
