#pragma once

// Shared test helpers for CLI export tool tests.
// Used by bar_feature_export_test.cpp and hybrid_model_tb_label_test.cpp.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace export_test_helpers {

// Path to the built bar_feature_export binary.
const std::string BINARY_PATH = "build/bar_feature_export";

// Run a shell command and capture stdout+stderr and exit code.
struct RunResult {
    int exit_code;
    std::string output;
};

inline RunResult run_command(const std::string& cmd) {
    RunResult result;
    std::string full_cmd = cmd + " 2>&1";
    FILE* pipe = popen(full_cmd.c_str(), "r");
    if (!pipe) {
        result.exit_code = -1;
        result.output = "popen failed";
        return result;
    }
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result.output += buffer;
    }
    int status = pclose(pipe);
    result.exit_code = WEXITSTATUS(status);
    return result;
}

// Parse a CSV line into individual column values.
inline std::vector<std::string> parse_csv_header(const std::string& line) {
    std::vector<std::string> cols;
    std::istringstream ss(line);
    std::string col;
    while (std::getline(ss, col, ',')) {
        while (!col.empty() && (col.back() == '\r' || col.back() == '\n'))
            col.pop_back();
        cols.push_back(col);
    }
    return cols;
}

// Read the first line of a file.
inline std::string read_first_line(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f.is_open()) std::getline(f, line);
    return line;
}

// Read all non-empty lines of a file.
inline std::vector<std::string> read_all_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

}  // namespace export_test_helpers
