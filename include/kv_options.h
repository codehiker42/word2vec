#ifndef _KV_OPTIONS_H
#define _KV_OPTIONS_H

#include <cstdint>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

using OpValueType =
    std::variant<std::monostate, bool, char, uint8_t, uint16_t, uint32_t,
                 uint64_t, float, std::string, std::filesystem::path>;

constexpr OpValueType ReturnTrue(const char* /*unused*/) { return true; }

constexpr OpValueType ConvertBool(const char* arg_val) {
  if (!arg_val) {
    return std::monostate{};
  }
  return *arg_val == '1' || *arg_val == 't' || *arg_val == 'T';
}

template <typename T>
constexpr T ReadFromStream(const char* char_val) {
  std::istringstream iss(char_val);
  T num_val;
  iss >> num_val;
  return num_val;
}

template <typename T>
constexpr OpValueType Convert(const char* arg_val) {
  if (!arg_val) {
    return std::monostate{};
  }
  return ReadFromStream<T>(arg_val);
}

constexpr std::function<OpValueType(const char*)> ConvertFuncFactory(
    const std::type_info& typeinfo, bool has_arg) {
  if (typeinfo == typeid(bool) && !has_arg) {
    return ReturnTrue;
  } else if (typeinfo == typeid(bool)) {
    return ConvertBool;
  } else if (typeinfo == typeid(char)) {
    return Convert<char>;
  } else if (typeinfo == typeid(bool)) {
    return ConvertBool;
  } else if (typeinfo == typeid(std::uint8_t)) {
    return Convert<std::uint8_t>;
  } else if (typeinfo == typeid(std::uint16_t)) {
    return Convert<std::uint16_t>;
  } else if (typeinfo == typeid(std::uint32_t)) {
    return Convert<std::uint32_t>;
  } else if (typeinfo == typeid(std::uint64_t)) {
    return Convert<std::uint64_t>;
  } else if (typeinfo == typeid(float)) {
    return Convert<float>;
  } else if (typeinfo == typeid(std::filesystem::path)) {
    return Convert<std::filesystem::path>;
  } else {  // std::string
    return Convert<std::string>;
  }
}

struct OptionDescriptionAndValue {
  template <typename T>
  OptionDescriptionAndValue(std::string name, char short_opt, bool has_arg,
                            std::string description,
                            const std::tuple<T, std::string>& default_val_tup)
      : name_(std::move(name)),
        short_opt_(short_opt),
        has_arg_(has_arg),
        description_(std::move(description)),
        default_value_desc_(std::get<1>(default_val_tup)),
        value_(std::get<0>(default_val_tup)),
        value_func_(ConvertFuncFactory(typeid(T), has_arg)) {}

  OptionDescriptionAndValue(std::string name, char short_opt, bool has_arg,
                            std::string description,
                            const std::type_info& type_info)
      : name_(std::move(name)),
        short_opt_(short_opt),
        has_arg_(has_arg),
        description_(std::move(description)),
        value_func_(ConvertFuncFactory(type_info, has_arg)) {}

  void Set(const char* arg) { value_ = value_func_(arg); }

  std::string name_;
  char short_opt_;
  bool has_arg_;
  std::string description_;
  std::string default_value_desc_;
  OpValueType value_;
  std::function<OpValueType(const char* arg)> value_func_;
};

class ConsoleOptionValues {
 public:
  ConsoleOptionValues() = delete;

  ConsoleOptionValues(
      int argc, char** argv,
      std::initializer_list<OptionDescriptionAndValue>&& op_values);

  std::string HelpMessage() const;

 protected:
  template <typename T>
  const T get(const std::string& name) const {
    return std::get<T>(value_map_.at(name).value_);
  }

  template <typename T>
  const std::optional<T> get_op(const std::string& name) const {
    const auto& op_var = value_map_.at(name).value_;
    if (op_var.index() == 0) {
      return {};
    } else {
      return {std::get<T>(op_var)};
    }
  }

  virtual bool HasMandatoryOptions() const = 0;

 private:
  std::vector<std::string> name_order_vec_;
  std::unordered_map<std::string, OptionDescriptionAndValue> value_map_;
};

#endif