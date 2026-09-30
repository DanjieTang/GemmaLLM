// Benchmark greedy generation of the Gemma text decoder in CUDA.
//
//   make && ./gemma_llm                          # random weights, sweep_config.yaml
//   ./gemma_llm --weights exported --verify      # weights and reference from Python
//
// Run ./gemma_llm --help for all options.
#include "common.cuh"
#include "model.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

const char* kUsage = R"(Usage: gemma_llm [options]
  --config PATH         Flat YAML with the architecture (default ../../sweep_config.yaml)
  --weights DIR         Load DIR/config.yaml and DIR/*.bin instead of random weights
  --verify              Compare with DIR/reference_tokens.txt and DIR/reference_logits.bin
  --total_tokens N      Sequence length to reach, prompt included (default 1024)
  --prompt_length N     Prompt tokens before generation (default 1, the BOS token)
  --warmup N            Untimed generations before measuring (default 1)
  --runs N              Timed generations (default 3)
  --seed N              Random weight seed (default 0)
  --no_graph            Launch each decode step's kernels directly, without a CUDA graph
  --<key> VALUE         Override an architecture setting, e.g. --max_context_length 2048
                        (num_layer, vocab_size, text_dim, projection_dim, expansion_factor,
                        head_dim, q_head, kv_head, theta, norm_eps, embedding_dtype)
)";

// Same deterministic prompt as benchmark_pytorch.py: BOS (2), then spread-out IDs.
std::vector<int> make_prompt(int length, int vocab_size) {
    std::vector<int> prompt(length);
    for (int i = 0; i < length; ++i)
        prompt[i] = static_cast<int>((i == 0 ? 2LL : 7919LL * i) % vocab_size);
    return prompt;
}

std::vector<int> read_reference_tokens(const std::string& path, int& prompt_length) {
    std::ifstream file(path);
    std::vector<int> tokens;
    if (!(file >> prompt_length)) throw std::runtime_error("Cannot read " + path);
    for (int token; file >> token;) tokens.push_back(token);
    if (prompt_length < 1 || int(tokens.size()) <= prompt_length)
        throw std::runtime_error(path + " must hold a prompt length and a longer token sequence");
    return tokens;
}

std::vector<float> read_float32(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != count * sizeof(float))
        throw std::runtime_error(path + ": expected " + std::to_string(count * sizeof(float)) + " bytes");
    std::vector<float> values(count);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return values;
}

// Returns false when the post-prompt logits disagree with PyTorch.
bool verify(GemmaLLM& model, const std::string& directory, bool use_graph) {
    int prompt_length = 0;
    const std::vector<int> expected = read_reference_tokens(directory + "/reference_tokens.txt", prompt_length);
    const std::vector<float> expected_logits =
        read_float32(directory + "/reference_logits.bin", model.config().vocab_size);
    const std::vector<int> prompt(expected.begin(), expected.begin() + prompt_length);
    std::vector<float> logits;
    const std::vector<int> tokens = model.generate(prompt, int(expected.size()), use_graph, &logits);

    constexpr float atol = 1e-3f, rtol = 1e-3f;
    float max_error = 0;
    size_t mismatches = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const float error = std::abs(logits[i] - expected_logits[i]);
        max_error = std::max(max_error, error);
        if (!std::isfinite(logits[i]) || error > atol + rtol * std::abs(expected_logits[i])) ++mismatches;
    }
    std::cout << "Logits after the prompt: max |CUDA - PyTorch| = " << max_error << ", "
              << logits.size() - mismatches << "/" << logits.size() << " within atol=" << atol
              << " rtol=" << rtol << " -> " << (mismatches ? "FAIL" : "PASS") << "\n";

    int matching = prompt_length;
    while (matching < int(tokens.size()) && tokens[matching] == expected[matching]) ++matching;
    const int generated = int(tokens.size()) - prompt_length;
    std::cout << "Greedy tokens: " << matching - prompt_length << "/" << generated
              << " generated tokens match PyTorch before the first difference\n";
    if (matching < int(tokens.size()))
        std::cout << "  (float32 rounding can flip a near-tie in argmax, after which sequences diverge)\n";
    return mismatches == 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ModelConfig config;
        std::string config_path = "../../sweep_config.yaml", weights;
        int total_tokens = 1024, prompt_length = 1, warmup = 1, runs = 3;
        uint64_t seed = 0;
        bool use_graph = true, check = false;
        std::vector<std::pair<std::string, std::string>> overrides;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(arg + " requires a value\n" + kUsage);
                return argv[++i];
            };
            if (arg == "--help" || arg == "-h") { std::cout << kUsage; return 0; }
            else if (arg == "--config") config_path = value();
            else if (arg == "--weights") weights = value();
            else if (arg == "--verify") check = true;
            else if (arg == "--total_tokens") total_tokens = std::stoi(value());
            else if (arg == "--prompt_length") prompt_length = std::stoi(value());
            else if (arg == "--warmup") warmup = std::stoi(value());
            else if (arg == "--runs") runs = std::stoi(value());
            else if (arg == "--seed") seed = std::stoull(value());
            else if (arg == "--no_graph") use_graph = false;
            else if (arg.rfind("--", 0) == 0) overrides.emplace_back(arg.substr(2), value());
            else throw std::runtime_error("Unexpected argument " + arg + "\n" + kUsage);
        }
        if (check && weights.empty()) throw std::runtime_error("--verify requires --weights DIR");
        if (prompt_length < 1 || warmup < 0 || runs < 0) throw std::runtime_error("Invalid run counts");

        // Exported weights carry their own architecture; otherwise use the sweep YAML.
        apply_yaml(config, weights.empty() ? config_path : weights + "/config.yaml");
        for (const auto& [key, text] : overrides)
            if (!apply_setting(config, key, text)) throw std::runtime_error("Unknown option --" + key);
        int reference_prompt = 0;
        if (check) {
            const auto reference = read_reference_tokens(weights + "/reference_tokens.txt", reference_prompt);
            config.max_context_length = std::max(config.max_context_length, int(reference.size()));
        }
        if (total_tokens > config.max_context_length) {
            std::cout << "Raising max_context_length from " << config.max_context_length << " to "
                      << total_tokens << " (RoPE table and KV cache size only)\n";
            config.max_context_length = total_tokens;
        }

        cudaDeviceProp properties{};
        int device = 0;
        CHECK_CUDA(cudaGetDevice(&device));
        CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
        GemmaLLM model(config, std::max(prompt_length, reference_prompt));
        std::cout << "GPU: " << properties.name << "\n"
                  << "Config: " << model.config().summary() << "\n"
                  << "Parameters: " << std::fixed << std::setprecision(1)
                  << model.parameter_count() / 1e6 << "M (embedding table "
                  << model.embedding_bytes() / 1e9 << " GB)\n";
        if (weights.empty()) {
            model.init_random(seed);
            std::cout << "Weights: random (seed " << seed << ")\n";
        } else {
            model.load_weights(weights);
            std::cout << "Weights: " << weights << "\n";
        }
        std::cout << "Decode: " << (use_graph ? "CUDA graph per token" : "direct kernel launches") << "\n";

        bool verified = true;
        if (check) verified = verify(model, weights, use_graph);
        if (runs == 0) return verified ? 0 : 1;

        const std::vector<int> prompt = make_prompt(prompt_length, model.config().vocab_size);
        for (int i = 0; i < warmup; ++i) model.generate(prompt, total_tokens, use_graph);
        std::vector<double> seconds;
        for (int run = 0; run < runs; ++run) {
            const auto start = std::chrono::steady_clock::now();
            model.generate(prompt, total_tokens, use_graph);  // Returns after the final token is on the host.
            seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
            std::cout << "Run " << run + 1 << "/" << runs << ": " << total_tokens << " tokens in "
                      << std::setprecision(1) << seconds.back() * 1e3 << " ms\n";
        }
        double mean = 0;
        for (double value : seconds) mean += value / runs;
        const int generated = total_tokens - prompt_length;
        std::cout << std::setprecision(1) << "Mean: " << mean * 1e3 << " ms to reach " << total_tokens
                  << " tokens (" << generated / mean << " generated tokens/s, "
                  << std::setprecision(3) << mean * 1e3 / generated << " ms/token)\n"
                  << std::setprecision(6) << "RESULT implementation=cuda total_tokens=" << total_tokens
                  << " mean_seconds=" << mean << "\n";
        return verified ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
