
#include <gtest/gtest.h>
#include <vector>
#include <string>
#include "register/op_def_registry.h"

namespace ops {
namespace {
class OpDefFactoryUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

class AddAscendC : public OpDef {
 public:
  AddAscendC(const char *name) : OpDef(name) {}
};

OP_ADD(AddAscendC, None);

TEST_F(OpDefFactoryUT, OpDefFactoryTest) {
  auto &ops = OpDefFactory::GetAllOp();
  for (auto &op : ops) {
    OpDef opDef = OpDefFactory::OpDefCreate(op.GetString());
    EXPECT_EQ(opDef.GetOpType(), "AddAscendC");
    EXPECT_EQ(opDef.GetWorkspaceFlag(), true);
    opDef.SetWorkspaceFlag(false);
    EXPECT_EQ(opDef.GetWorkspaceFlag(), false);
  }
  EXPECT_EQ(ops.size(), 1);
}

}  // namespace
}  // namespace ops
