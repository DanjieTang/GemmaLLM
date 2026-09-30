// Parse the flat YAML subset used by sweep_config.yaml and exported configs.
#include "common.cuh"

#include <fstream>
#include <sstream>

namespace {

std::string trim(const std::string& text) {
    const char* space = " \t\r\n";
    size_t begin = text.find_first_not_of(space);
    if (begin == std::string::npos) return "";
    return text.substr(begin, text.find_last_not_of(space) - begin + 1);
}

bool is_null(const std::string& value) {
    return value == "null" || value == "~" || value.empty();
}

int parse_int(const std::string& key, const std::string& value) {
    size_t used = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size() || parsed < 0 || parsed > 1'000'000'000)
        throw std::runtime_error(key + ": expected a nonnegative integer, got '" + value + "'");
    return static_cast<int>(parsed);
}

float parse_float(const std::string& key, const std::string& value) {
    size_t used = 0;
    float parsed = 0;
    try {
        parsed = std::stof(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size())
        throw std::runtime_error(key + ": expected a number, got '" + value + "'");
    return parsed;
}

}  // namespace

bool apply_setting(ModelConfig& c, const std::string& key, const std::string& raw_value) {
    const std::string value = trim(raw_value);
    // Nullable integers mirror the Python defaults (None -> derived later).
    auto nullable_int = [&](int& field) { field = is_null(value) ? 0 : parse_int(key, value); };
    if (key == "num_layer") c.num_layer = parse_int(key, value);
    else if (key == "vocab_size") c.vocab_size = parse_int(key, value);
    else if (key == "text_dim") c.text_dim = parse_int(key, value);
    else if (key == "projection_dim") nullable_int(c.projection_dim);
    else if (key == "expansion_factor") c.expansion_factor = parse_int(key, value);
    else if (key == "head_dim") c.head_dim = parse_int(key, value);
    else if (key == "q_head") nullable_int(c.q_head);
    else if (key == "kv_head") nullable_int(c.kv_head);
    else if (key == "theta") c.theta = parse_float(key, value);
    else if (key == "max_context_length") c.max_context_length = parse_int(key, value);
    else if (key == "norm_eps") c.norm_eps = parse_float(key, value);
    else if (key == "embedding_dtype") {
        if (value == "bfloat16") c.embedding_dtype = EmbeddingDtype::BFloat16;
        else if (value == "float32") c.embedding_dtype = EmbeddingDtype::Float32;
        else throw std::runtime_error("embedding_dtype must be bfloat16 or float32");
    } else if (key == "use_moe") {
        if (value == "true" || value == "True")
            throw std::runtime_error("use_moe: true is not implemented in the CUDA model");
    } else {
        return false;
    }
    return true;
}

void apply_yaml(ModelConfig& config, const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open config " + path);
    std::string line;
    while (std::getline(file, line)) {
        // Only unindented `key: value` lines; list items and comments are skipped.
        if (line.empty() || line[0] == ' ' || line[0] == '\t' || line[0] == '#' || line[0] == '-')
            continue;
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string value = line.substr(colon + 1);
        size_t comment = value.find(" #");
        if (comment != std::string::npos) value.resize(comment);
        apply_setting(config, trim(line.substr(0, colon)), value);
    }
}

void ModelConfig::finalize() {
    if (num_layer <= 0 || vocab_size <= 0 || text_dim <= 0 || head_dim <= 0 ||
        expansion_factor <= 0 || max_context_length <= 0 || !(norm_eps > 0) || !(theta > 0))
        throw std::runtime_error("Model sizes, theta, and norm_eps must be positive");
    const int hidden = hidden_dim();
    if (q_head == 0) q_head = hidden / head_dim;
    if (kv_head == 0) kv_head = hidden / head_dim;
    if (head_dim != 32 && head_dim != 64 && head_dim != 128 && head_dim != 256)
        throw std::runtime_error("head_dim must be 32, 64, 128, or 256 for the attention kernel");
    // Attention reshapes [q_head, head_dim] back to hidden_dim, so they must agree.
    if (q_head <= 0 || kv_head <= 0 || q_head * head_dim != hidden || q_head % kv_head)
        throw std::runtime_error("Require q_head * head_dim == hidden_dim and q_head divisible by kv_head");
}

std::string ModelConfig::summary() const {
    std::ostringstream out;
    out << "layers=" << num_layer << " vocab=" << vocab_size << " text_dim=" << text_dim
        << " hidden=" << hidden_dim() << " ffn=" << ffn_dim()
        << " heads(q/kv)=" << q_head << "/" << kv_head << "x" << head_dim
        << " theta=" << theta << " max_context=" << max_context_length
        << " embeddings=" << (embedding_dtype == EmbeddingDtype::BFloat16 ? "bfloat16" : "float32")
        << (has_projection() ? " (projected)" : "");
    return out.str();
}
