#pragma once

#include <boost/program_options.hpp>

#include "3rd_party/yaml-cpp/yaml.h"
#include "common/logging.h"

namespace marian {

class Config {
  public:
    Config(int argc, char** argv, bool validate = true) {
      Logger info{stderrLogger("info", "[%Y-%m-%d %T] %v")};
      Logger config{stderrLogger("config", "[%Y-%m-%d %T] [config] %v")};
      Logger memory{stderrLogger("memory", "[%Y-%m-%d %T] [memory] %v")};
      Logger data{stderrLogger("data", "[%Y-%m-%d %T] [data] %v")};
      Logger valid{stderrLogger("valid", "[%Y-%m-%d %T] [valid] %v")};

      addOptions(argc, argv, validate);

      log();
    }

    bool has(const std::string& key) const;

    YAML::Node get(const std::string& key) const;

    template <typename T>
    T get(const std::string& key) const {
      return config_[key].as<T>();
    }

    const YAML::Node& get() const;
    YAML::Node& get();

    YAML::Node operator[](const std::string& key) const {
      return get(key);
    }

    void addOptions(int argc, char** argv, bool validate);
    void log();
    void validate() const;

    template <class OStream>
    friend OStream& operator<<(OStream& out, const Config& config) {
      out << config.config_;
      return out;
    }

  private:
    std::string inputPath;
    YAML::Node config_;
};

}