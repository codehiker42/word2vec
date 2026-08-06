#include "kv_options.h"

#include <getopt.h>
#include <unistd.h>

#include <memory>
#include <sstream>

ConsoleOptionValues::ConsoleOptionValues(
    int argc, char** argv,
    std::initializer_list<OptionDescriptionAndValue>&& op_values) {
  auto long_opt_arr = std::make_unique<option[]>(op_values.size() + 1);
  long_opt_arr[op_values.size()] = {nullptr, 0, nullptr, 0};

  std::string short_op_str;
  std::unordered_map<int, std::string> short_op_map;
  for (auto i = 0; i < op_values.size(); ++i) {
    auto& op_val = *(op_values.begin() + i);
    name_order_vec_.emplace_back(op_val.name_);
    value_map_.emplace(op_val.name_, op_val);
    long_opt_arr[i] = {op_val.name_.c_str(), op_val.has_arg_, nullptr,
                       op_val.short_opt_};

    if (op_val.short_opt_) {
      short_op_str += static_cast<char>(op_val.short_opt_);
      if (op_val.has_arg_) {
        short_op_str += ':';
      }
      short_op_map[op_val.short_opt_] = op_val.name_;
    }
  }

  for (int opt, opt_ind = 0;
       (opt = getopt_long(argc, argv, short_op_str.c_str(), long_opt_arr.get(),
                          &opt_ind)) != -1;) {
    auto name = opt == 0 ? long_opt_arr[opt_ind].name : short_op_map[opt];
    value_map_.at(name).Set(optarg);
  }
}

std::string ConsoleOptionValues::HelpMessage() const {
  std::ostringstream oss;
  oss << "WORD VECTOR estimation toolkit v 0.1c\n";  // TODO CMake ver & types
  oss << "Parameters for training:\n";

  for (const auto& op_name : name_order_vec_) {
    const auto& op_value = value_map_.at(op_name);
    oss << "--" << op_name;
    if (op_value.short_opt_ != 0) {
      oss << ", -" << op_value.short_opt_;
    }
    auto colon_pos = op_value.description_.find_first_of(';');
    if (op_value.has_arg_ && colon_pos != std::string::npos) {
      oss << " " << op_value.description_.substr(0, colon_pos);
    }
    oss << "\n\t"
        << (colon_pos == std::string::npos
                ? op_value.description_
                : op_value.description_.substr(colon_pos + 1));
    if (!op_value.default_value_desc_.empty()) {
      oss << ", default value: " << op_value.default_value_desc_;
    }
    oss << "\n";
  }
  return oss.str();
}
