#include "include/json.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr float EPSILON = 1e-5f;
constexpr int EMBEDDING_DIMENSION = 768;
constexpr int NUM_LAYERS = 12;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBEDDING_DIMENSION / NUM_HEADS;
constexpr int VOCAB_SIZE = 50257;
constexpr int CONTEXT_LENGTH = 1024;

using tensor = std::vector<float>;

struct TransformerInput {
  tensor qWeights, kWeights, vWeights;
  tensor qBiases, kBiases, vBiases;

  tensor l1Weights, l2Weights;
  tensor l1Biases, l2Biases;

  tensor oWeights, oBiases;

  tensor lnAttnWeights, lnAttnBiases;
  tensor lnMlpWeights, lnMlpBiases;
};

struct gptWeights {
  tensor embeddingWeights, positionalEmbeddingWeights;
  std::vector<TransformerInput> transformerWeights;
  tensor finalWeights, finalBiases;

  gptWeights() : transformerWeights(NUM_LAYERS) {
    auto readFromFileStream = [](std::vector<float> &vec, std::ifstream &stream,
                                 int maxCount = 0) {
      float input;
      if (maxCount > 0) {
        while (maxCount-- > 0 && stream >> input)
          vec.push_back(input);
      } else {
        while (stream >> input)
          vec.push_back(input);
      }
    };

    auto load = [&](tensor &dest, const std::string &path) {
      std::ifstream f(path);
      if (!f) {
        std::cerr << "failed to open " << path << "\n";
        std::exit(1);
      }
      readFromFileStream(dest, f);
    };

    // HF Conv1D c_attn weight is (768, 2304) = (in, out), fused as [Q|K|V]
    auto loadFusedQkv = [&](TransformerInput &layer,
                            const std::string &prefix) {
      const int nEmb = EMBEDDING_DIMENSION;
      const int nFused = 3 * nEmb;

      {
        std::ifstream f(prefix + ".attn.c_attn.bias.txt");
        readFromFileStream(layer.qBiases, f, nEmb);
        readFromFileStream(layer.kBiases, f, nEmb);
        readFromFileStream(layer.vBiases, f, nEmb);
      }

      {
        std::ifstream f(prefix + ".attn.c_attn.weight.txt");
        float x;
        layer.qWeights.reserve(nEmb * nEmb);
        layer.kWeights.reserve(nEmb * nEmb);
        layer.vWeights.reserve(nEmb * nEmb);
        for (int row = 0; row < nEmb; row++) {
          for (int col = 0; col < nFused; col++) {
            f >> x;
            if (col < nEmb)
              layer.qWeights.push_back(x);
            else if (col < 2 * nEmb)
              layer.kWeights.push_back(x);
            else
              layer.vWeights.push_back(x);
          }
        }
      }
    };

    std::cerr << "loading embedding weights...\n";
    load(embeddingWeights, "../weights/transformer.wte.weight.txt");
    load(positionalEmbeddingWeights, "../weights/transformer.wpe.weight.txt");

    for (int i = 0; i < NUM_LAYERS; i++) {
      auto &layer = transformerWeights[i];
      std::string prefix = "../weights/transformer.h." + std::to_string(i);
      std::cerr << "loading layer " << i << "...\n";

      load(layer.lnAttnWeights, prefix + ".ln_1.weight.txt");
      load(layer.lnAttnBiases, prefix + ".ln_1.bias.txt");

      loadFusedQkv(layer, prefix);

      load(layer.oWeights, prefix + ".attn.c_proj.weight.txt");
      load(layer.oBiases, prefix + ".attn.c_proj.bias.txt");

      load(layer.lnMlpWeights, prefix + ".ln_2.weight.txt");
      load(layer.lnMlpBiases, prefix + ".ln_2.bias.txt");

      load(layer.l1Weights, prefix + ".mlp.c_fc.weight.txt");
      load(layer.l1Biases, prefix + ".mlp.c_fc.bias.txt");
      load(layer.l2Weights, prefix + ".mlp.c_proj.weight.txt");
      load(layer.l2Biases, prefix + ".mlp.c_proj.bias.txt");
    }

    load(finalWeights, "../weights/transformer.ln_f.weight.txt");
    load(finalBiases, "../weights/transformer.ln_f.bias.txt");
    std::cerr << "weights loaded.\n";
  }
};

float gelu(float x) {

  return 0.5f * x *
         (1.0f + std::tanh(std::sqrt(2.0f / static_cast<float>(M_PI)) *
                           (x + 0.044715f * x * x * x)));
}

std::vector<float> matMul(std::span<const float> a, int an, int am,
                          std::span<const float> b, int bn, int bm) {
  assert(am == bn);
  std::vector<float> result(an * bm, 0.0f);

  for (int i = 0; i < an; i++) {
    for (int k = 0; k < bn; k++) {
      float aik = a[i * am + k];
      for (int j = 0; j < bm; j++)
        result[i * bm + j] += aik * b[k * bm + j];
    }
  }
  return result;
}

std::vector<float> transpose(std::span<const float> a, int n, int m) {
  std::vector<float> result(n * m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      result[j * n + i] = a[i * m + j];
  return result;
}

float dotProduct(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size());
  float result = 0.0f;
  for (size_t i = 0; i < a.size(); i++)
    result += a[i] * b[i];
  return result;
}

tensor addVectors(std::span<const float> a, std::span<const float> b) {
  assert(a.size() == b.size());
  tensor result(a.begin(), a.end());
  for (size_t i = 0; i < a.size(); i++)
    result[i] += b[i];
  return result;
}

tensor layerNorm(std::span<const float> ogEmbeddings,
                 std::span<const float> weights,
                 std::span<const float> biases) {
  assert(ogEmbeddings.size() == weights.size());
  assert(ogEmbeddings.size() == biases.size());

  const int n = static_cast<int>(ogEmbeddings.size());
  float mean = 0.0f;
  for (float v : ogEmbeddings)
    mean += v;
  mean /= static_cast<float>(n);

  float variance = 0.0f;
  for (float v : ogEmbeddings) {
    float d = v - mean;
    variance += d * d;
  }
  variance /= static_cast<float>(n);

  float invStd = 1.0f / std::sqrt(variance + EPSILON);

  tensor output(n);
  for (int i = 0; i < n; i++) {
    float normalized = (ogEmbeddings[i] - mean) * invStd;
    output[i] = normalized * weights[i] + biases[i];
  }
  return output;
}

tensor softmax(std::span<const float> input) {
  const int n = static_cast<int>(input.size());
  float mx = *std::max_element(input.begin(), input.end());

  tensor output(n);
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    output[i] = std::exp(input[i] - mx);
    sum += output[i];
  }
  for (float &v : output)
    v /= sum;
  return output;
}

// HF Conv1D: y = x @ W + b, W stored (in, out) row-major
tensor forwardPass(std::span<const float> weights,
                   std::span<const float> biases, std::span<const float> inputs,
                   bool useGelu = false) {
  const int outDim = static_cast<int>(biases.size());
  const int inDim = static_cast<int>(inputs.size());
  assert(static_cast<int>(weights.size()) == inDim * outDim);

  tensor output(biases.begin(), biases.end());
  for (int j = 0; j < inDim; j++) {
    float x = inputs[j];
    for (int i = 0; i < outDim; i++)
      output[i] += weights[j * outDim + i] * x;
  }
  if (useGelu) {
    for (float &v : output)
      v = gelu(v);
  }
  return output;
}

tensor attention(std::span<const float> embeddings, int numTokens, int embedDim,
                 int headDim, int headIdx, std::span<const float> qWeights,
                 std::span<const float> kWeights,
                 std::span<const float> vWeights,
                 std::span<const float> qBiases, std::span<const float> kBiases,
                 std::span<const float> vBiases) {
  tensor qProjections(numTokens * headDim);
  tensor kProjections(numTokens * headDim);
  tensor vProjections(numTokens * headDim);

  const int outBase = headIdx * headDim;

  for (int t = 0; t < numTokens; t++) {
    auto tokenEmb = embeddings.subspan(t * embedDim, embedDim);
    for (int d = 0; d < headDim; d++) {
      const int out = outBase + d;
      float q = qBiases[out];
      float k = kBiases[out];
      float v = vBiases[out];
      for (int i = 0; i < embedDim; i++) {
        q += tokenEmb[i] * qWeights[i * embedDim + out];
        k += tokenEmb[i] * kWeights[i * embedDim + out];
        v += tokenEmb[i] * vWeights[i * embedDim + out];
      }
      qProjections[t * headDim + d] = q;
      kProjections[t * headDim + d] = k;
      vProjections[t * headDim + d] = v;
    }
  }

  auto kT = transpose(kProjections, numTokens, headDim);
  auto scores =
      matMul(qProjections, numTokens, headDim, kT, headDim, numTokens);

  const float scale = std::sqrt(static_cast<float>(headDim));
  for (float &s : scores)
    s /= scale;

  // causal mask: disallow attending to future tokens
  for (int i = 0; i < numTokens; i++)
    for (int j = i + 1; j < numTokens; j++)
      scores[i * numTokens + j] = -std::numeric_limits<float>::infinity();

  for (int i = 0; i < numTokens; i++) {
    auto row = std::span<const float>(scores).subspan(i * numTokens, numTokens);
    auto softrow = softmax(row);
    for (int j = 0; j < numTokens; j++)
      scores[i * numTokens + j] = softrow[j];
  }

  return matMul(scores, numTokens, numTokens, vProjections, numTokens, headDim);
}

tensor multiHeadAttention(
    int numTokens, int embedDim, int heads, int headDim,
    std::span<const float> embeddings, std::span<const float> qWeights,
    std::span<const float> kWeights, std::span<const float> vWeights,
    std::span<const float> qBiases, std::span<const float> kBiases,
    std::span<const float> vBiases, std::span<const float> oWeights,
    std::span<const float> oBiases) {
  tensor concat(numTokens * embedDim, 0.0f);

  for (int h = 0; h < heads; h++) {
    auto headOut =
        attention(embeddings, numTokens, embedDim, headDim, h, qWeights,
                  kWeights, vWeights, qBiases, kBiases, vBiases);

    for (int i = 0; i < numTokens; i++)
      for (int j = 0; j < headDim; j++)
        concat[i * embedDim + h * headDim + j] = headOut[i * headDim + j];
  }

  // oWeights is HF Conv1D (embedDim, embedDim) = (in, out)
  auto projected =
      matMul(concat, numTokens, embedDim, oWeights, embedDim, embedDim);
  for (int i = 0; i < numTokens; i++)
    for (int j = 0; j < embedDim; j++)
      projected[i * embedDim + j] += oBiases[j];

  return projected;
}

tensor mlp(int numTokens, int dimensions, std::span<const float> embeddings,
           std::span<const float> l1Weights, std::span<const float> l1Biases,
           std::span<const float> l2Weights, std::span<const float> l2Biases) {
  tensor result(numTokens * dimensions);
  for (int i = 0; i < numTokens; i++) {
    auto tokenEmb = embeddings.subspan(i * dimensions, dimensions);
    auto hiddenOut = forwardPass(l1Weights, l1Biases, tokenEmb, true);
    auto out = forwardPass(l2Weights, l2Biases, hiddenOut);
    for (int j = 0; j < dimensions; j++)
      result[i * dimensions + j] = out[j];
  }
  return result;
}

tensor transformer(const TransformerInput &input, int numTokens,
                   const tensor &embeddings) {
  tensor normedAttn(numTokens * EMBEDDING_DIMENSION);
  for (int i = 0; i < numTokens; i++) {
    auto tokenEmbedding =
        std::span<const float>(embeddings)
            .subspan(i * EMBEDDING_DIMENSION, EMBEDDING_DIMENSION);
    auto normed =
        layerNorm(tokenEmbedding, input.lnAttnWeights, input.lnAttnBiases);
    for (int j = 0; j < EMBEDDING_DIMENSION; j++)
      normedAttn[i * EMBEDDING_DIMENSION + j] = normed[j];
  }

  auto attentionResult = multiHeadAttention(
      numTokens, EMBEDDING_DIMENSION, NUM_HEADS, HEAD_DIM, normedAttn,
      input.qWeights, input.kWeights, input.vWeights, input.qBiases,
      input.kBiases, input.vBiases, input.oWeights, input.oBiases);

  tensor residual1 = addVectors(embeddings, attentionResult);

  tensor normedMlp(numTokens * EMBEDDING_DIMENSION);
  for (int i = 0; i < numTokens; i++) {
    auto token = std::span<const float>(residual1).subspan(
        i * EMBEDDING_DIMENSION, EMBEDDING_DIMENSION);
    auto normed = layerNorm(token, input.lnMlpWeights, input.lnMlpBiases);
    for (int j = 0; j < EMBEDDING_DIMENSION; j++)
      normedMlp[i * EMBEDDING_DIMENSION + j] = normed[j];
  }

  auto mlpResult =
      mlp(numTokens, EMBEDDING_DIMENSION, normedMlp, input.l1Weights,
          input.l1Biases, input.l2Weights, input.l2Biases);

  return addVectors(residual1, mlpResult);
}

// --- tokenizer (GPT-2 byte-level BPE) ---

std::vector<std::string> gpt2Tokens;
std::unordered_map<std::string, int> gpt2TokenToTokenId;
std::map<std::pair<std::string, std::string>, int> merges;
std::unordered_map<uint8_t, char32_t> byteEncoder;
std::unordered_map<char32_t, uint8_t> byteDecoder;

std::string utf8Encode(char32_t cp) {
  std::string out;
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
  return out;
}

std::u32string utf8Decode(const std::string &s) {
  std::u32string out;
  size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp = 0;
    int len = 0;
    if ((c & 0x80) == 0) {
      cp = c;
      len = 1;
    } else if ((c & 0xE0) == 0xC0) {
      cp = c & 0x1F;
      len = 2;
    } else if ((c & 0xF0) == 0xE0) {
      cp = c & 0x0F;
      len = 3;
    } else {
      cp = c & 0x07;
      len = 4;
    }
    for (int k = 1; k < len; k++) {
      if (i + k >= s.size())
        break;
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

void buildByteEncoder() {
  std::vector<int> bs;
  for (int i = 33; i <= 126; i++)
    bs.push_back(i);
  for (int i = 161; i <= 172; i++)
    bs.push_back(i);
  for (int i = 174; i <= 255; i++)
    bs.push_back(i);

  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; b++) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      n++;
    }
  }
  for (size_t i = 0; i < bs.size(); i++) {
    byteEncoder[static_cast<uint8_t>(bs[i])] = static_cast<char32_t>(cs[i]);
    byteDecoder[static_cast<char32_t>(cs[i])] = static_cast<uint8_t>(bs[i]);
  }
}

bool isLetter(char32_t c) {
  return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z') ||
         (c >= 0x00C0 && c <= 0x02AF) || (c >= 0x0400 && c <= 0x04FF);
}

bool isNumber(char32_t c) { return c >= U'0' && c <= U'9'; }

bool isWhitespace(char32_t c) {
  return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f';
}

// Approximate GPT-2 regex pretokenizer for common Latin text
std::vector<std::string> preTokenize(const std::string &text) {
  std::u32string s = utf8Decode(text);
  std::vector<std::string> pieces;
  size_t i = 0;
  auto endsWith = [&](const char32_t *lit) {
    size_t n = 0;
    while (lit[n])
      n++;
    if (i + n > s.size())
      return false;
    for (size_t k = 0; k < n; k++)
      if (s[i + k] != lit[k])
        return false;
    return true;
  };

  while (i < s.size()) {
    size_t start = i;
    if (endsWith(U"'s") || endsWith(U"'t") || endsWith(U"'m") ||
        endsWith(U"'d")) {
      i += 2;
    } else if (endsWith(U"'re") || endsWith(U"'ve") || endsWith(U"'ll")) {
      i += 3;
    } else if (i + 1 < s.size() && s[i] == U' ' && isLetter(s[i + 1])) {
      i++;
      while (i < s.size() && isLetter(s[i]))
        i++;
    } else if (isLetter(s[i])) {
      while (i < s.size() && isLetter(s[i]))
        i++;
    } else if (i + 1 < s.size() && s[i] == U' ' && isNumber(s[i + 1])) {
      i++;
      while (i < s.size() && isNumber(s[i]))
        i++;
    } else if (isNumber(s[i])) {
      while (i < s.size() && isNumber(s[i]))
        i++;
    } else if (isWhitespace(s[i])) {
      while (i < s.size() && isWhitespace(s[i]))
        i++;
      // GPT-2: keep trailing whitespace split when more non-space follows
      if (i < s.size() && !isWhitespace(s[i]) && i > start + 1) {
        // move last whitespace to next match — match HF behaviour loosely
        i--;
      }
    } else {
      if (i < s.size() && s[i] == U' ')
        i++;
      while (i < s.size() && !isWhitespace(s[i]) && !isLetter(s[i]) &&
             !isNumber(s[i]))
        i++;
      if (i == start)
        i++;
    }

    std::string piece;
    for (size_t k = start; k < i; k++)
      piece += utf8Encode(s[k]);
    if (!piece.empty())
      pieces.push_back(piece);
  }
  return pieces;
}

std::vector<std::string> bpe(const std::string &token) {
  if (token.empty())
    return {};
  if (gpt2TokenToTokenId.count(token))
    return {token};

  std::vector<std::string> word;
  std::u32string chars = utf8Decode(token);
  for (char32_t c : chars)
    word.push_back(utf8Encode(c));

  if (word.size() == 1)
    return word;

  while (true) {
    int bestRank = std::numeric_limits<int>::max();
    int bestIdx = -1;
    for (size_t i = 0; i + 1 < word.size(); i++) {
      auto it = merges.find({word[i], word[i + 1]});
      if (it != merges.end() && it->second < bestRank) {
        bestRank = it->second;
        bestIdx = static_cast<int>(i);
      }
    }
    if (bestIdx < 0)
      break;

    std::vector<std::string> next;
    next.reserve(word.size());
    for (size_t i = 0; i < word.size();) {
      if (static_cast<int>(i) == bestIdx) {
        next.push_back(word[i] + word[i + 1]);
        i += 2;
      } else {
        next.push_back(word[i]);
        i++;
      }
    }
    word.swap(next);
    if (word.size() == 1)
      break;
  }
  return word;
}

std::string getTokenFromTokenId(int tokenId) {
  if (tokenId < 0 || tokenId >= static_cast<int>(gpt2Tokens.size()))
    return "";
  return gpt2Tokens[tokenId];
}

void parseMerges(const nlohmann::json &model) {
  merges.clear();
  const auto &m = model.at("merges");
  int rank = 0;
  for (const auto &entry : m) {
    if (entry.is_array() && entry.size() == 2) {
      merges[{entry[0].get<std::string>(), entry[1].get<std::string>()}] =
          rank++;
    } else if (entry.is_string()) {
      std::string s = entry.get<std::string>();
      auto sp = s.find(' ');
      if (sp != std::string::npos)
        merges[{s.substr(0, sp), s.substr(sp + 1)}] = rank++;
    }
  }
}

void loadVocab() {
  using json = nlohmann::json;
  std::ifstream f("../weights/tokenizer/tokenizer.json");
  if (!f) {
    std::cerr << "failed to open tokenizer.json\n";
    std::exit(1);
  }
  const json data = json::parse(f);
  const json &model = data.at("model");

  buildByteEncoder();

  gpt2Tokens.assign(VOCAB_SIZE, "");
  gpt2TokenToTokenId.clear();
  for (auto it = model.at("vocab").begin(); it != model.at("vocab").end();
       ++it) {
    const std::string token = it.key();
    const int id = it.value().get<int>();
    if (id >= 0 && id < VOCAB_SIZE) {
      gpt2Tokens[id] = token;
      gpt2TokenToTokenId[token] = id;
    }
  }

  parseMerges(model);
  std::cerr << "tokenizer loaded: vocab=" << gpt2TokenToTokenId.size()
            << " merges=" << merges.size() << "\n";
}

std::vector<int> tokenize(const std::string &text) {
  std::vector<int> ids;
  for (const std::string &piece : preTokenize(text)) {
    // byte-level encode: map raw bytes -> unicode chars used by GPT-2 vocab
    std::string encoded;
    for (unsigned char b : piece)
      encoded += utf8Encode(byteEncoder[b]);

    for (const std::string &tok : bpe(encoded)) {
      auto it = gpt2TokenToTokenId.find(tok);
      if (it == gpt2TokenToTokenId.end()) {
        std::cerr << "unknown token in vocab\n";
        std::exit(1);
      }
      ids.push_back(it->second);
    }
  }
  return ids;
}

std::string detokenize(const std::vector<int> &ids) {
  std::string encoded;
  for (int id : ids)
    encoded += getTokenFromTokenId(id);

  std::u32string chars = utf8Decode(encoded);
  std::string out;
  out.reserve(chars.size());
  for (char32_t c : chars) {
    auto it = byteDecoder.find(c);
    if (it != byteDecoder.end())
      out.push_back(static_cast<char>(it->second));
  }
  return out;
}

tensor getTokenEmbedding(const gptWeights &weights,
                         const std::vector<int> &tokenIds) {
  const int n = static_cast<int>(tokenIds.size());
  tensor emb(n * EMBEDDING_DIMENSION);
  for (int t = 0; t < n; t++) {
    const int id = tokenIds[t];
    assert(id >= 0 && id < VOCAB_SIZE);
    assert(t < CONTEXT_LENGTH);
    for (int d = 0; d < EMBEDDING_DIMENSION; d++) {
      emb[t * EMBEDDING_DIMENSION + d] =
          weights.embeddingWeights[id * EMBEDDING_DIMENSION + d] +
          weights.positionalEmbeddingWeights[t * EMBEDDING_DIMENSION + d];
    }
  }
  return emb;
}

tensor forwardModel(const gptWeights &weights,
                    const std::vector<int> &tokenIds) {
  tensor hidden = getTokenEmbedding(weights, tokenIds);
  const int numTokens = static_cast<int>(tokenIds.size());

  for (int layer = 0; layer < NUM_LAYERS; layer++)
    hidden = transformer(weights.transformerWeights[layer], numTokens, hidden);

  // final layer norm
  tensor normed(numTokens * EMBEDDING_DIMENSION);
  for (int i = 0; i < numTokens; i++) {
    auto token = std::span<const float>(hidden).subspan(i * EMBEDDING_DIMENSION,
                                                        EMBEDDING_DIMENSION);
    auto n = layerNorm(token, weights.finalWeights, weights.finalBiases);
    for (int j = 0; j < EMBEDDING_DIMENSION; j++)
      normed[i * EMBEDDING_DIMENSION + j] = n[j];
  }
  return normed;
}

tensor logitsForLastToken(const gptWeights &weights, const tensor &normedHidden,
                          int numTokens) {
  // tied LM head: logits = hidden @ wte^T
  auto last =
      std::span<const float>(normedHidden)
          .subspan((numTokens - 1) * EMBEDDING_DIMENSION, EMBEDDING_DIMENSION);
  tensor logits(VOCAB_SIZE);
  for (int v = 0; v < VOCAB_SIZE; v++) {
    float sum = 0.0f;
    for (int d = 0; d < EMBEDDING_DIMENSION; d++)
      sum += last[d] * weights.embeddingWeights[v * EMBEDDING_DIMENSION + d];
    logits[v] = sum;
  }
  return logits;
}

int sampleTopK(const tensor &logits, int k, float temperature,
               std::mt19937 &rng) {
  struct Pair {
    float logit;
    int id;
  };
  std::vector<Pair> scored;
  scored.reserve(logits.size());
  for (int i = 0; i < static_cast<int>(logits.size()); i++)
    scored.push_back({logits[i], i});

  if (k <= 0 || k > static_cast<int>(scored.size()))
    k = static_cast<int>(scored.size());

  std::partial_sort(
      scored.begin(), scored.begin() + k, scored.end(),
      [](const Pair &a, const Pair &b) { return a.logit > b.logit; });
  scored.resize(k);

  float mx = scored[0].logit;
  tensor probs(k);
  float sum = 0.0f;
  const float temp = std::max(temperature, 1e-5f);
  for (int i = 0; i < k; i++) {
    probs[i] = std::exp((scored[i].logit - mx) / temp);
    sum += probs[i];
  }
  for (float &p : probs)
    p /= sum;

  std::discrete_distribution<int> dist(probs.begin(), probs.end());
  return scored[dist(rng)].id;
}

int gpt(const gptWeights &weights, const std::vector<int> &tokenIds, int topK,
        float temperature, std::mt19937 &rng) {
  auto hidden = forwardModel(weights, tokenIds);
  auto logits =
      logitsForLastToken(weights, hidden, static_cast<int>(tokenIds.size()));
  return sampleTopK(logits, topK, temperature, rng);
}

int main() {
  constexpr int topK = 40;
  constexpr float temperature = 0.8f;

  loadVocab();
  gptWeights weights;
  std::mt19937 rng{std::random_device{}()};

  while (true) {
    std::string prompt;
    int maxNewTokens = 0;

    std::cout << "prompt (empty to quit): " << std::flush;
    if (!std::getline(std::cin, prompt) || prompt.empty())
      break;

    std::cout << "num tokens: " << std::flush;
    if (!(std::cin >> maxNewTokens) || maxNewTokens <= 0) {
      std::cerr << "need a positive token count\n";
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    auto tokenIds = tokenize(prompt);
    std::cout << detokenize(tokenIds) << std::flush;
    for (int t = 0; t < maxNewTokens; t++) {
      if (static_cast<int>(tokenIds.size()) >= CONTEXT_LENGTH)
        break;
      int next = gpt(weights, tokenIds, topK, temperature, rng);
      tokenIds.push_back(next);
      std::cout << detokenize({next}) << std::flush;
      if (next == 50256) // <|endoftext|>
        break;
    }
    std::cout << "\n\n";
  }
  return 0;
}