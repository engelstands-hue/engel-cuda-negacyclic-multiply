// GENERATED harness. The platform may overwrite this file before measuring.
#include "fherma.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
namespace fs = std::filesystem;
namespace {
std::string slurp(const fs::path& at) {
    std::ifstream file(at, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read " + at.string());
    std::ostringstream out; out << file.rdbuf(); return out.str();
}
double number(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const size_t at = text.find(quoted);
    if (at == std::string::npos) throw std::runtime_error("no " + key);
    return std::strtod(text.c_str() + text.find(':', at + quoted.size()) + 1, nullptr);
}
std::string numbered(size_t i) { std::ostringstream out; out << std::setw(6) << std::setfill('0') << i; return out.str(); }
template <class T>
fherma::Tensor<T> read_tensor(const fs::path& where, const std::string& name, std::vector<int64_t> shape) {
    fherma::Tensor<T> tensor; tensor.shape = std::move(shape);
    const std::string raw = slurp(where / (name + ".bin"));
    const size_t count = static_cast<size_t>(tensor.count());
    if (raw.size() != count * sizeof(T)) throw std::runtime_error(name + " size");
    tensor.data.resize(count); std::memcpy(tensor.data.data(), raw.data(), raw.size()); return tensor;
}
template <class T>
void write_tensor(const fs::path& where, const std::string& name, const fherma::Tensor<T>& tensor) {
    std::ofstream out(where / (name + ".bin"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(tensor.data.data()), static_cast<std::streamsize>(tensor.data.size() * sizeof(T)));
}
void report(const fs::path& out, double init_s, const std::vector<std::string>& cases) {
    std::ofstream file(out / "results.json");
    file << std::fixed << std::setprecision(9) << "{\"init_s\":" << init_s << ",\"cases\":[";
    for (size_t i = 0; i < cases.size(); ++i) { if (i) file << ","; file << cases[i]; }
    file << "]}";
}

}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: solution <point directory>\n"; return 2; }
    const fs::path root = argv[1];
    const std::string manifest = slurp(root / "manifest.json");
    fherma::Point p{};
    p.N = static_cast<uint32_t>(number(manifest, "N"));
    p.W = static_cast<uint32_t>(number(manifest, "W"));
    p.L = static_cast<uint32_t>(number(manifest, "L"));
    p.q = read_tensor<uint32_t>(root / "point", "q", {static_cast<int64_t>(p.L)});
    const size_t total = static_cast<size_t>(number(manifest, "cases"));
    const fs::path answers_root = root / "out";
    fs::create_directories(answers_root);
    const auto setup = std::chrono::steady_clock::now();
    void* state = fherma_init(p);
    const std::chrono::duration<double> init_took = std::chrono::steady_clock::now() - setup;
    std::vector<std::string> cases;
    report(answers_root, init_took.count(), cases);
    for (size_t i = 0; i < total; ++i) {
        const fs::path where = root / "cases" / numbered(i);
        const fs::path answers = answers_root / numbered(i);
        try {
            fherma::Inputs in{};
            in.a = read_tensor<uint32_t>(where, "a", {static_cast<int64_t>(p.N), static_cast<int64_t>(p.L)});
            in.b = read_tensor<uint32_t>(where, "b", {static_cast<int64_t>(p.N), static_cast<int64_t>(p.L)});
            const auto started = std::chrono::steady_clock::now();
            fherma::Outputs answer = fherma_run(state, in);
            const std::chrono::duration<double> took = std::chrono::steady_clock::now() - started;
            fs::create_directories(answers);
            write_tensor(answers, "c", answer.c);
            std::ostringstream row;
            row << std::fixed << std::setprecision(9) << "{\"i\":" << i << ",\"seconds\":" << took.count() << ",\"status\":\"ok\"}";
            cases.push_back(row.str());
        } catch (const std::exception& failure) {
            cases.push_back(std::string("{\"i\":") + std::to_string(i) + ",\"seconds\":null,\"status\":\"crashed\"}");
        }
        report(answers_root, init_took.count(), cases);
    }
    fherma_free(state);
    report(answers_root, init_took.count(), cases);
    return 0;
}
