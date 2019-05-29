#include <LightGBM/tree.h>

#include <LightGBM/utils/threading.h>
#include <LightGBM/utils/common.h>

#include <LightGBM/dataset.h>
#include <LightGBM/feature.h>

#include <sstream>
#include <unordered_map>
#include <functional>
#include <vector>
#include <string>
#include <memory>

namespace LightGBM {

std::vector<std::function<bool(unsigned int, unsigned int)>> Tree::inner_decision_funs = 
          {Tree::NumericalDecision<unsigned int>, Tree::CategoricalDecision<unsigned int> };
std::vector<std::function<bool(float, float)>> Tree::decision_funs = 
          { Tree::NumericalDecision<float>, Tree::CategoricalDecision<float> };


Tree::Tree(int max_leaves)
  :max_leaves_(max_leaves) {

  num_leaves_ = 0;
  left_child_ = std::vector<int>(max_leaves_ - 1);
  right_child_ = std::vector<int>(max_leaves_ - 1);
  split_feature_ = std::vector<int>(max_leaves_ - 1);
  split_feature_real_ = std::vector<int>(max_leaves_ - 1);
  threshold_in_bin_ = std::vector<unsigned int>(max_leaves_ - 1);
  threshold_ = std::vector<float>(max_leaves_ - 1);
  decision_type_ = std::vector<int8_t>(max_leaves_ - 1);
  split_gain_ = std::vector<float>(max_leaves_ - 1);
  leaf_parent_ = std::vector<int>(max_leaves_);
  leaf_value_ = std::vector<float>(max_leaves_);
  leaf_count_ = std::vector<data_size_t>(max_leaves_);
  internal_value_ = std::vector<float>(max_leaves_ - 1);
  internal_count_ = std::vector<data_size_t>(max_leaves_ - 1);
  leaf_depth_ = std::vector<int>(max_leaves_);
  // root is in the depth 1
  leaf_depth_[0] = 1;
  num_leaves_ = 1;
  leaf_parent_[0] = -1;
}
Tree::~Tree() {

}

int Tree::Split(int leaf, int feature, BinType bin_type, unsigned int threshold_bin, int real_feature,
    float threshold_float, float left_value,
    float right_value, data_size_t left_cnt, data_size_t right_cnt, float gain) {
  int new_node_idx = num_leaves_ - 1;
  // update parent info
  int parent = leaf_parent_[leaf];
  if (parent >= 0) {
    // if cur node is left child
    if (left_child_[parent] == ~leaf) {
      left_child_[parent] = new_node_idx;
    } else {
      right_child_[parent] = new_node_idx;
    }
  }
  // add new node
  split_feature_[new_node_idx] = feature;
  split_feature_real_[new_node_idx] = real_feature;
  threshold_in_bin_[new_node_idx] = threshold_bin;
  threshold_[new_node_idx] = threshold_float;
  if (bin_type == BinType::NumericalBin) {
    decision_type_[new_node_idx] = 0;
  } else {
    decision_type_[new_node_idx] = 1;
  }
  split_gain_[new_node_idx] = gain;
  // add two new leaves
  left_child_[new_node_idx] = ~leaf;
  right_child_[new_node_idx] = ~num_leaves_;
  // update new leaves
  leaf_parent_[leaf] = new_node_idx;
  leaf_parent_[num_leaves_] = new_node_idx;
  // save current leaf value to internal node before change
  internal_value_[new_node_idx] = leaf_value_[leaf];
  internal_count_[new_node_idx] = left_cnt + right_cnt;
  leaf_value_[leaf] = left_value;
  leaf_count_[leaf] = left_cnt;
  leaf_value_[num_leaves_] = right_value;
  leaf_count_[num_leaves_] = right_cnt;
  // update leaf depth
  leaf_depth_[num_leaves_] = leaf_depth_[leaf] + 1;
  leaf_depth_[leaf]++;

  ++num_leaves_;
  return num_leaves_ - 1;
}

void Tree::AddPredictionToScore(const Dataset* data, data_size_t num_data, score_t* score) const {
  Threading::For<data_size_t>(0, num_data, [this, data, score](int, data_size_t start, data_size_t end) {
    std::vector<std::unique_ptr<BinIterator>> iterators(data->num_features());
    for (int i = 0; i < data->num_features(); ++i) {
      iterators[i].reset(data->FeatureAt(i)->bin_data()->GetIterator(start));
    }
    for (data_size_t i = start; i < end; ++i) {
      score[i] += static_cast<score_t>(leaf_value_[GetLeaf(iterators, i)]);
    }
  });
}

void Tree::AddPredictionToScore(const Dataset* data, const data_size_t* used_data_indices,
                                             data_size_t num_data, score_t* score) const {
  Threading::For<data_size_t>(0, num_data,
      [this, data, used_data_indices, score](int, data_size_t start, data_size_t end) {
    std::vector<std::unique_ptr<BinIterator>> iterators(data->num_features());
    for (int i = 0; i < data->num_features(); ++i) {
      iterators[i].reset(data->FeatureAt(i)->bin_data()->GetIterator(used_data_indices[start]));
    }
    for (data_size_t i = start; i < end; ++i) {
      score[used_data_indices[i]] += static_cast<score_t>(leaf_value_[GetLeaf(iterators, used_data_indices[i])]);
    }
  });
}

std::string Tree::ToString() {
  std::stringstream ss;
  ss << "num_leaves=" << num_leaves_ << std::endl;
  ss << "split_feature="
    << Common::ArrayToString<int>(split_feature_real_, num_leaves_ - 1, ' ') << std::endl;
  ss << "split_gain="
    << Common::ArrayToString<float>(split_gain_, num_leaves_ - 1, ' ') << std::endl;
  ss << "threshold="
    << Common::ArrayToString<float>(threshold_, num_leaves_ - 1, ' ') << std::endl;
  ss << "decision_type="
    << Common::ArrayToString<int>(Common::ArrayCast<int8_t, int>(decision_type_), num_leaves_ - 1, ' ') << std::endl;
  ss << "left_child="
    << Common::ArrayToString<int>(left_child_, num_leaves_ - 1, ' ') << std::endl;
  ss << "right_child="
    << Common::ArrayToString<int>(right_child_, num_leaves_ - 1, ' ') << std::endl;
  ss << "leaf_parent="
    << Common::ArrayToString<int>(leaf_parent_, num_leaves_, ' ') << std::endl;
  ss << "leaf_value="
    << Common::ArrayToString<float>(leaf_value_, num_leaves_, ' ') << std::endl;
  ss << "leaf_count="
    << Common::ArrayToString<data_size_t>(leaf_count_, num_leaves_, ' ') << std::endl;
  ss << "internal_value="
    << Common::ArrayToString<float>(internal_value_, num_leaves_ - 1, ' ') << std::endl;
  ss << "internal_count="
    << Common::ArrayToString<data_size_t>(internal_count_, num_leaves_ - 1, ' ') << std::endl;
  ss << std::endl;
  return ss.str();
}
std::string Tree::ToStringFicha2(const char *filename, int tree_num, int idx, int loop, int max_num_leaves, int shift_bit)
{
	std::stringstream ss;
	std::string model_tmp;

	std::vector<int> split_index;
	split_index.push_back(0);
	split_index.push_back(left_child_[0]);
	split_index.push_back(right_child_[0]);


	int depthmax = log2(max_num_leaves);
	int i = 1;
	bool build = true;
	while (build)
	{
		// if get every leafnode
		int *leafnode = new int[num_leaves_];
		int leafnodenum = 0;
		for (int k = 1; k <= num_leaves_; k++)
		{
			leafnode[k - 1] = 0;
			for (int j = 0; j < split_index.size(); j++)
			{
				if (split_index[j] == -k)
				{
					leafnode[k - 1] = k;
				}
			}
		}
		for (int k = 1; k <= num_leaves_; k++)
		{
			if (leafnode[k - 1] != 0)
			{
				leafnodenum++;
			}
		}
		delete[]leafnode;

		//reach all the leaves 
		if (leafnodenum == num_leaves_)//&&split_index.size() == fulltree)
		{
			// cal fulltree size
			int fulltree = std::pow(2, depthmax + 1) - 1;
			if (split_index.size() == fulltree)
			{
				build = false;
				//split_index[0] = fulltree;
			}
		}
		if (build == false)break;
		if (split_index[i] >= 0)
		{
			split_index.push_back(left_child_[split_index[i]]);
			split_index.push_back(right_child_[split_index[i]]);
		}
		else if (split_index[i] < 0)
		{
			split_index.push_back(split_index[i]);
			split_index.push_back(split_index[i]);
		}
		i++;
	}
	
	if (loop == 5)
	{
		if (idx == 0)
		{
			model_tmp = "int " + std::string(filename) + "_index[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}

		std::vector<int> feature;
		feature.push_back(split_feature_real_[0]);
		for (int index = 1; index < split_index.size(); index++)
		{
			if (split_index[index]>=0)
			{
				feature.push_back(split_feature_real_[split_index[index]]);
			}
			if (split_index[index] < 0)
			{
				
				int depth = log2(index+1);
				if (depth == depthmax)
				{
					feature.push_back(split_index[index]);
				}
				else if (depth < depthmax)
				{
					int tmp = index;
					int parent = 0;
					while (1)
					{
						if (split_index[tmp] >= 0)break;
						if (tmp % 2 == 0)
							parent = (tmp - 2) / 2;
						else if (tmp % 2 == 1)
							parent = (tmp - 1) / 2;
						tmp = parent;
					}
					
					feature.push_back(split_feature_real_[split_index[tmp]]);
				}
			}
		}
		ss << Common::ArrayToStringFicha(feature, feature.size(), ", ", tree_num, idx);

		int vectorsize = split_feature_real_.size();
		if (idx <= tree_num - 1)
		{
			//for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
			//	ss << ", ";
			if (idx != tree_num - 1)
				ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;

		//split_index.clear();
		feature.clear();
	}
	if (loop == 6)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_level[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}

		std::vector<short> level;
		level.push_back(ceil(threshold_[0]));

		for (int index = 1; index < split_index.size(); index++)
		{
			if (split_index[index]>0)
			{
				level.push_back(ceil(threshold_[split_index[index]]));
			}
			if (split_index[index] < 0)
			{
				int depth = log2(index + 1);
				if (depth == depthmax)
				{
					//printf("thresh1:%f\n", threshold_[abs(split_index[index])]);
					level.push_back(ceil(threshold_[abs(split_index[index])]));
				}
				else if (depth < depthmax)
				{
					int tmp = index;
					int parent = 0;
					while (1)
					{
						if (split_index[tmp] >= 0)break;
						if (tmp % 2 == 0)
							parent = (tmp - 2) / 2;
						else if (tmp % 2 == 1)
							parent = (tmp - 1) / 2;
						tmp = parent;
						//printf("tmp :%d\n", tmp);
					}
					//printf("thresh2:%f\n", threshold_[split_index[tmp]]);
					level.push_back(ceil(threshold_[split_index[tmp]]));
				}
			}
		}

		ss << Common::ArrayToStringFicha(level, level.size(), ", ", tree_num, idx);

		int	vectorsize = threshold_.size();
		if (idx <= tree_num - 1)
		{
			//for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
			//	ss << ", ";
			if (idx != tree_num - 1)
				ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;

		//split_index.clear();
		level.clear();
	}
	if (loop == 7)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_value[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}

		std::vector<float> value;
		value.push_back(leaf_value_[0]);
		for (int index = 1; index < split_index.size(); index++)
		{
			int tmp = split_index[index];

			if (tmp < 0)
				tmp = std::abs(tmp)-1;

			value.push_back(leaf_value_[tmp]);
		}

		ss << Common::ArrayToStringFicha(value, value.size(), ", ", tree_num, idx, shift_bit);

		int	vectorsize = leaf_value_.size();
		if (idx <= tree_num - 1)
		{
			//for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
			//	ss << ", ";
			if (idx != tree_num - 1)
				ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;

		//split_index.clear();
		value.clear();
	}
	split_index.clear();
	return ss.str();
}
std::string Tree::ToStringFicha(const char *filename, int tree_num, int idx, int loop, int max_num_leaves, int shift_bit) {
	std::stringstream ss;
	std::string model_tmp;

	if (loop == 0)
	{
		if (idx == 0)
		{
			model_tmp = "int "+std::string(filename) + "_index[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}
		
		ss << Common::ArrayToStringFicha(split_feature_real_, num_leaves_ - 1, ", ", tree_num, idx);

		int vectorsize = split_feature_real_.size();
		if (idx <= tree_num - 1)
		{
			for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
			    ss << ", 0";
			if (idx != tree_num - 1)
			ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;
	}
	
	if (loop == 1)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_level[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}
		size_t len = std::min((size_t)num_leaves_ - 1, threshold_.size());
		std::vector<short>threshold_short;

		for (size_t i = 0; i <len ; ++i) {
			
			short tmp = ceil(threshold_[i]);
			threshold_short.push_back(tmp);
			
		}
		
		ss << Common::ArrayToStringFicha(threshold_short, num_leaves_ - 1, ", ", tree_num, idx);

		int vectorsize = threshold_short.size();
		if (idx <= tree_num - 1)
		{
			for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
				ss << ", 0";
			if (idx != tree_num - 1)
			ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;
	}

	if (loop == 2)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_value[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}
		ss << Common::ArrayToStringFicha(leaf_value_, num_leaves_, ", ", tree_num, idx, shift_bit);
		int vectorsize = leaf_value_.size();
		if (idx <= tree_num - 1)
		{
			for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_); i++)
				ss << ", 0";
			if (idx != tree_num - 1)
			ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;
	}

	if (loop == 3)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_left_child[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}
		ss << Common::ArrayToStringFicha(left_child_, num_leaves_ - 1, ", ", tree_num, idx);
		int vectorsize = left_child_.size();
		if (idx <= tree_num - 1)
		{
			for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
				ss << ", 0";
			if (idx != tree_num - 1)
			ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;
    }
	if (loop == 4)
	{
		if (idx == 0)
		{
			model_tmp = "short " + std::string(filename) + "_right_child[";
			ss << model_tmp << max_num_leaves*tree_num << "] = { ";
		}
		ss << Common::ArrayToStringFicha(right_child_, num_leaves_ - 1, ", ", tree_num, idx);
		int vectorsize = right_child_.size();
		if (idx <= tree_num - 1)
		{
			for (int i = 0; i < max_num_leaves - std::min(vectorsize, num_leaves_ - 1); i++)
				ss << ", 0";
			if (idx != tree_num - 1)
			ss << ", ";
		}
		if (idx == tree_num - 1)	ss << " };" << std::endl;
	}
	
	return ss.str();
}
std::string Tree::ToJSON() {
  std::stringstream ss;

  ss << "\"num_leaves\":" << num_leaves_ << "," << std::endl;

  ss << "\"tree_structure\":" << NodeToJSON(0) << std::endl;

  return ss.str();
}

std::string Tree::NodeToJSON(int index) {
  std::stringstream ss;

  if (index >= 0) {
    // non-leaf
    ss << "{" << std::endl;
    ss << "\"split_index\":" << index << "," << std::endl;
    ss << "\"split_feature\":" << split_feature_real_[index] << "," << std::endl;
    ss << "\"split_gain\":" << split_gain_[index] << "," << std::endl;
    ss << "\"threshold\":" << threshold_[index] << "," << std::endl;
    ss << "\"decision_type\":\"" << Tree::GetDecisionTypeName(decision_type_[index]) << "\"," << std::endl;
    ss << "\"internal_value\":" << internal_value_[index] << "," << std::endl;
    ss << "\"internal_count\":" << internal_count_[index] << "," << std::endl;
    ss << "\"left_child\":" << NodeToJSON(left_child_[index]) << "," << std::endl;
    ss << "\"right_child\":" << NodeToJSON(right_child_[index]) << std::endl;
    ss << "}";
  } else {
    // leaf
    index = ~index;
    ss << "{" << std::endl;
    ss << "\"leaf_index\":" << index << "," << std::endl;
    ss << "\"leaf_parent\":" << leaf_parent_[index] << "," << std::endl;
    ss << "\"leaf_value\":" << leaf_value_[index] << "," << std::endl;
    ss << "\"leaf_count\":" << leaf_count_[index] << std::endl;
    ss << "}";
  }

  return ss.str();
}

Tree::Tree(const std::string& str) {
  std::vector<std::string> lines = Common::Split(str.c_str(), '\n');
  std::unordered_map<std::string, std::string> key_vals;
  for (const std::string& line : lines) {
    std::vector<std::string> tmp_strs = Common::Split(line.c_str(), '=');
    if (tmp_strs.size() == 2) {
      std::string key = Common::Trim(tmp_strs[0]);
      std::string val = Common::Trim(tmp_strs[1]);
      if (key.size() > 0 && val.size() > 0) {
        key_vals[key] = val;
      }
    }
  }
  if (key_vals.count("num_leaves") <= 0 || key_vals.count("split_feature") <= 0
    || key_vals.count("split_gain") <= 0 || key_vals.count("threshold") <= 0
    || key_vals.count("left_child") <= 0 || key_vals.count("right_child") <= 0
    || key_vals.count("leaf_parent") <= 0 || key_vals.count("leaf_value") <= 0
    || key_vals.count("internal_value") <= 0 || key_vals.count("internal_count") <= 0
    || key_vals.count("leaf_count") <= 0 || key_vals.count("decision_type") <= 0
    ) {
    Log::Fatal("Tree model string format error");
  }

  Common::Atoi(key_vals["num_leaves"].c_str(), &num_leaves_);

  left_child_ = Common::StringToArray<int>(key_vals["left_child"], ' ', num_leaves_ - 1);
  right_child_ = Common::StringToArray<int>(key_vals["right_child"], ' ', num_leaves_ - 1);
  split_feature_real_ = Common::StringToArray<int>(key_vals["split_feature"], ' ', num_leaves_ - 1);
  threshold_ = Common::StringToArray<float>(key_vals["threshold"], ' ', num_leaves_ - 1);
  split_gain_ = Common::StringToArray<float>(key_vals["split_gain"], ' ', num_leaves_ - 1);
  internal_count_ = Common::StringToArray<data_size_t>(key_vals["internal_count"], ' ', num_leaves_ - 1);
  internal_value_ = Common::StringToArray<float>(key_vals["internal_value"], ' ', num_leaves_ - 1);
  decision_type_ = Common::StringToArray<int8_t>(key_vals["decision_type"], ' ', num_leaves_ - 1);

  leaf_count_ = Common::StringToArray<data_size_t>(key_vals["leaf_count"], ' ', num_leaves_);
  leaf_parent_ = Common::StringToArray<int>(key_vals["leaf_parent"], ' ', num_leaves_);
  //leaf_value_ = Common::StringToArray<float>(key_vals["leaf_value"], ' ', num_leaves_);
  leaf_value_ = Common::StringToArray<float>(key_vals["leaf_value"], ' ', num_leaves_);
}

}  // namespace LightGBM
