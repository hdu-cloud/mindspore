#include <cinttypes>
#include <ctime>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "DistributedIR/graph.hpp"
#include "DistributedIR/node.hpp"
#include "DistributedIR/op.hpp"

using namespace framework;

class adapter {

    public:
	framework::Graph hdu_graph;
	std::map<std::string, std::vector<std::vector<std::int64_t> >> strategy;
	std::map<std::string, std::int64_t> op_stage_id;

	void ConvertMSGraphToHDUGraph(framework::NodeBase hdu_node){
        	this->hdu_graph.AddNode(hdu_node);
    	}
    	void ConvertMSNodeToHDUNode(const std::string node_name, const std::string op,
                            const std::string device, const std::vector<std::string> inputs,
                            const std::map<std::string, std::string> attrs) {
        	framework::NodeBase hdu_node;
        	hdu_node.Device(device);
        	hdu_node.Name(node_name);
        	hdu_node.Op(op);
        	hdu_node.Inputs(inputs);
        	hdu_node.Attrs(attrs);
        	ConvertMSGraphToHDUGraph(hdu_node);
    	}

	std::vector<std::vector<std::int64_t> > GetHDUStrategy(string op) {
                return this->strategy[op];
        }

	void SetHDUStrategy(std::string op, std::vector<std::vector<std::int64_t> > stra) {
                this->strategy.insert(std::pair<std::string, std::vector<std::vector<std::int64_t> >>(op, stra));

        }

	std::int64_t GetStageId(std::string op) {
                return this->op_stage_id[op];

        }

        void SetStageId(std::string op, std::int64_t stageId) {
                this->op_stage_id.insert(std::pair<std::string, std::int64_t>(op, stageId));
        }
};


