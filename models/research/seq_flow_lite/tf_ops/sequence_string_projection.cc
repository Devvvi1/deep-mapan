/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tf_ops/projection_normalizer_util.h"  // seq_flow_lite
#include "tf_ops/projection_tokenizer_util.h"  // seq_flow_lite
#include "tf_ops/projection_util.h"  // seq_flow_lite
#include "tf_ops/text_distorter.h"  // seq_flow_lite

using ::tensorflow::int32;
using ::tensorflow::int64;
using ::tensorflow::OpKernel;
using ::tensorflow::OpKernelConstruction;
using ::tensorflow::OpKernelContext;
using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::TensorShapeUtils;
using ::tensorflow::uint64;
using ::tensorflow::errors::InvalidArgument;

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;

constexpr char kBeginTokenTSP[] = "<BOS>";
constexpr char kEndTokenTSP[] = "<EOS>";

float* AllocateTensor(OpKernelContext* ctx, const std::string& tensor_name,
                      const TensorShape& tensor_shape) {
  Tensor* tensor = nullptr;
  auto status = ctx->allocate_output(tensor_name, tensor_shape, &tensor);
  if (!TF_PREDICT_TRUE(status.ok())) {
    ctx->CtxFailureWithWarning(__FILE__, __LINE__, status);
    return nullptr;
  }
  return &tensor->flat<float>()(0);
}

// OpKernel for the sequence string projection op.
class SequenceStringProjectionOp : public OpKernel {
 public:
  explicit SequenceStringProjectionOp(OpKernelConstruction* context)
      : OpKernel(context), philox_(171), generator_(&philox_) {
    OP_REQUIRES_OK(context, context->GetAttr("feature_size", &feature_size_));
    std::string hashtype;
    OP_REQUIRES_OK(context, context->GetAttr("hashtype", &hashtype));
    hasher_ =
        absl::WrapUnique<Hasher>(Hasher::CreateHasher(feature_size_, hashtype));
    CHECK(hasher_);
    float distortion_probability = 0.0;
    OP_REQUIRES_OK(context, context->GetAttr("distortion_probability",
                                             &distortion_probability));
    text_distorter_ = absl::make_unique<TextDistorter>(distortion_probability);
    OP_REQUIRES_OK(context,
                   context->GetAttr("split_on_space", &split_on_space_));
    OP_REQUIRES_OK(context, context->GetAttr("max_splits", &max_splits_));
    OP_REQUIRES_OK(context, context->GetAttr("vocabulary", &vocabulary_));
    bool add_bos_tag;
    OP_REQUIRES_OK(context, context->GetAttr("add_bos_tag", &add_bos_tag));
    bos_tag_ = add_bos_tag ? 1 : 0;
    bool add_eos_tag;
    OP_REQUIRES_OK(context, context->GetAttr("add_eos_tag", &add_eos_tag));
    eos_tag_ = add_eos_tag ? 1 : 0;
    // When word_novelty_bits is set to a positive integer, the last feature
    // generated by the op captures the token frequency.
    OP_REQUIRES_OK(context,
                   context->GetAttr("word_novelty_bits", &word_novelty_bits_));
    CHECK_GE(word_novelty_bits_, 0);
    CHECK_LE(word_novelty_bits_, 7);
    if (word_novelty_bits_ != 0) {
      CHECK_GE(feature_size_, 1);
    }
    // When doc_size_levels is set to a positive integer, the second to last
    // feature generated by the op is derived from the log of the document
    // size.
    OP_REQUIRES_OK(context,
                   context->GetAttr("doc_size_levels", &doc_size_levels_));
    CHECK_GE(doc_size_levels_, 0);
    CHECK_LE(doc_size_levels_, 16);
    if (doc_size_levels_ != 0) {
      CHECK_GE(feature_size_, 2);
    }
    word_novelty_offset_ = 1.0f / (1 << word_novelty_bits_);
    bool exclude_nonalphaspace_unicodes;
    OP_REQUIRES_OK(context, context->GetAttr("exclude_nonalphaspace_unicodes",
                                             &exclude_nonalphaspace_unicodes));
    if (!vocabulary_.empty()) {
      CHECK(!exclude_nonalphaspace_unicodes);
    }
    unicode_handler_ = absl::make_unique<ProjectionUnicodeHandler>(
        vocabulary_, exclude_nonalphaspace_unicodes);
    vocabulary_size_ = unicode_handler_->NumberOfValidUnicodes();

    bool normalize_repetition;
    OP_REQUIRES_OK(context, context->GetAttr("normalize_repetition",
                                             &normalize_repetition));
    std::string separators;
    OP_REQUIRES_OK(context, context->GetAttr("token_separators", &separators));
    if (!separators.empty() || normalize_repetition) {
      projection_normalizer_ = absl::make_unique<ProjectionNormalizer>(
          separators, normalize_repetition);
    }

    OP_REQUIRES_OK(context, context->GetAttr("add_first_cap_feature",
                                             &add_first_cap_feature_));
    CHECK_GE(add_first_cap_feature_, 0.0);
    CHECK_LE(add_first_cap_feature_, 1.0);
    if (add_first_cap_feature_ > 0.0) {
      CHECK_GE(feature_size_, 3);
    }

    OP_REQUIRES_OK(context, context->GetAttr("add_all_caps_feature",
                                             &add_all_caps_feature_));
    CHECK_GE(add_all_caps_feature_, 0.0);
    CHECK_LE(add_all_caps_feature_, 1.0);
    if (add_all_caps_feature_ > 0.0) {
      CHECK_GE(feature_size_, 4);
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(input_tensor->shape()),
                InvalidArgument("input must be a vector, got shape: ",
                                input_tensor->shape().DebugString()));

    auto input_vec = input_tensor->vec<::tensorflow::tstring>();
    const int64 batch_size = input_vec.dimension(0);
    std::vector<std::vector<std::pair<const char*, size_t>>> words_batches;
    int64 max_seq_len = 0;
    words_batches.reserve(batch_size);
    std::vector<std::string> normalized_input_vec(batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      std::vector<std::pair<const char*, size_t>> words;
      if (projection_normalizer_ == nullptr) {
        words =
            unicode_handler_->Tokenize(input_vec(i).data(), input_vec(i).size(),
                                       split_on_space_, max_splits_);
      } else {
        normalized_input_vec[i] = projection_normalizer_->Normalize(
            input_vec(i).data(), input_vec(i).size(), SIZE_MAX);
        words = unicode_handler_->Tokenize(normalized_input_vec[i],
                                           split_on_space_, max_splits_);
      }
      const int64 seq_len =
          static_cast<int64>(bos_tag_ + words.size() + eos_tag_);
      CHECK_GT(seq_len, 0)
          << "Projection models expect input text to have at-least one valid "
             "token. If empty text is a valid input for your model, please set "
             "add_bos_tag to true.";
      max_seq_len = std::max(max_seq_len, seq_len);
      words_batches.emplace_back(std::move(words));
    }

    auto projection =
        AllocateTensor(ctx, "projection",
                       TensorShape({batch_size, max_seq_len, feature_size_}));
    AllocateTensor(ctx, "dummy_output", TensorShape({1}));
    auto sequence_length =
        AllocateTensor(ctx, "sequence_length", TensorShape({batch_size}));
    if (!projection || !sequence_length) {
      LOG(ERROR) << "Unable to create buffer!";
      return;
    }

    const float mapping_table[4] = {0, 1, -1, 0};
    const int increment = 32;
    std::vector<uint64_t> hash_codes;
    absl::flat_hash_map<uint64, int> word_counter;
    for (int64 i = 0; i < batch_size; ++i) {
      word_counter.clear();
      const int64 num_tokens = words_batches[i].size();
      sequence_length[i] = bos_tag_ + num_tokens + eos_tag_;
      int64 offset0 = i * max_seq_len * feature_size_;
      // Calculate doc_size_feature in [0, infinity)
      float doc_size_feature =
          (doc_size_levels_ != 0)
              ? std::log2(static_cast<float>(num_tokens)) / doc_size_levels_
              : 0.0f;
      // Rescale doc_size_feature to [-1, 1].
      doc_size_feature = std::min(doc_size_feature, 1.0f) * 2.0f - 1.0f;
      for (int64 j = -bos_tag_; j < num_tokens + eos_tag_; ++j) {
        std::string word;
        bool first_cap = false;
        bool all_caps = false;
        if (j < 0) {
          // Use a special tag for begin of sentence.
          word = kBeginTokenTSP;
        } else if (j < num_tokens) {
          auto uword = icu::UnicodeString::fromUTF8(
              unicode_handler_->LowerCaseUTF8WithSupportedUnicodes(
                  words_batches[i][j], &first_cap, &all_caps));
          word = text_distorter_->DistortText(&uword);
        } else {
          // Use a special tag for end of sentence.
          CHECK_EQ(eos_tag_, 1);
          word = kEndTokenTSP;
        }
        hasher_->GetHashCodes(word, hash_codes);
        for (int hindex = 0, k = 0; hindex < hash_codes.size(); hindex++) {
          auto hash = hash_codes[hindex];
          for (int kmax = std::min(k + increment, feature_size_); k < kmax;) {
            projection[offset0 + k++] = mapping_table[hash & 0x3];
            hash >>= 2;
          }
        }
        if (word_novelty_bits_ != 0 && !hash_codes.empty()) {
          const auto word_hash = hash_codes[0];
          projection[offset0 + feature_size_ - kWordNoveltyOffset] =
              std::min((word_counter[word_hash]++ * word_novelty_offset_),
                       1.0f) *
                  2.0f -
              1.0f;
        }
        if (doc_size_levels_ != 0) {
          projection[offset0 + feature_size_ - kDocSizeOffset] =
              doc_size_feature;
        }
        if (add_first_cap_feature_ > 0.0f) {
          if (generator_.RandFloat() <= add_first_cap_feature_) {
            projection[offset0 + feature_size_ - kFirstCapOffset] =
                first_cap ? 1.0 : -1.0;
          } else {
            projection[offset0 + feature_size_ - kFirstCapOffset] = 0.0;
          }
        }
        if (add_all_caps_feature_ > 0.0f) {
          if (generator_.RandFloat() <= add_all_caps_feature_) {
            projection[offset0 + feature_size_ - kAllCapsOffset] =
                all_caps ? 1.0 : -1.0;
          } else {
            projection[offset0 + feature_size_ - kAllCapsOffset] = 0.0;
          }
        }
        offset0 += feature_size_;
      }
      const int pending = (max_seq_len - (bos_tag_ + num_tokens + eos_tag_));
      memset(projection + offset0, 0, pending * feature_size_ * sizeof(float));
    }
  }

 private:
  // Objects used for random number generator.
  tensorflow::random::PhiloxRandom philox_;
  tensorflow::random::SimplePhilox generator_;

  // Dimensionality of the ternary vector for each token in the text.
  int32 feature_size_;
  // An object used to hash tokens in the text.
  std::unique_ptr<Hasher> hasher_;
  // An object used for distorting text before projection.
  std::unique_ptr<TextDistorter> text_distorter_;
  // An object used for manipulating unicode in the text. It performs tasks such
  // as retaining only whitelisted unicodes in the text tokens and lowercasing
  // them.
  std::unique_ptr<ProjectionUnicodeHandler> unicode_handler_;
  // An object used for normalizing tokens in the text. This performs tasks
  // such as identifying repeated characters and replace them with a single
  // instance.
  std::unique_ptr<ProjectionNormalizer> projection_normalizer_;
  // Character whitelist used by the projection operator.
  std::string vocabulary_;
  // Size of the character whitelist.
  int vocabulary_size_;
  // Maximum number of splits allowed in the text. The number of tokens in text
  // post segmentation will be utmost max_splits_ + 1.
  int32 max_splits_;
  // A flag that indicates how to segment text. When true text is segmented by
  // space. Otherwise it is segmented on unicode boundaries.
  bool split_on_space_;
  // When true include an end of sentence token in the projection.
  int eos_tag_;
  // When true include a begin of sentence token in the projection.
  int bos_tag_;
  // Number of bits used to capture word novelty. See tensorflow op
  // documentation below for details.
  int word_novelty_bits_;
  // Number of levels used to capture document size. See tensorflow op
  // documentation below for details.
  int doc_size_levels_;
  // Distance between levels used for word novelty.
  float word_novelty_offset_;
  // Adds boolean feature to indicate first_cap text with the below probability.
  float add_first_cap_feature_;
  // Adds boolean feature to indicate all_cap text with the below probability.
  float add_all_caps_feature_;
};

REGISTER_KERNEL_BUILDER(
    Name("SequenceStringProjection").Device(::tensorflow::DEVICE_CPU),
    SequenceStringProjectionOp);

REGISTER_OP("SequenceStringProjection")
    .Input("input: string")
    .Output("projection: float32")
    .Output("dummy_output: float32")
    .Output("sequence_length: float32")
    .Attr("feature_size: int")
    .Attr("distortion_probability: float = 0.0")
    .Attr("vocabulary: string = ''")
    .Attr("hashtype: string = 'murmur'")
    .Attr("max_splits: int = -1")
    .Attr("exclude_nonalphaspace_unicodes: bool = False")
    .Attr("add_bos_tag: bool = False")
    .Attr("add_eos_tag: bool = True")
    .Attr("add_first_cap_feature: float = 0.0")
    .Attr("add_all_caps_feature: float = 0.0")
    .Attr("word_novelty_bits: int = 0")
    .Attr("doc_size_levels: int = 0")
    .Attr("split_on_space: bool = True")
    .Attr("token_separators: string = ''")
    .Attr("normalize_repetition: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      DimensionHandle size;

      int32 feature_size;
      TF_RETURN_IF_ERROR(c->GetAttr("feature_size", &feature_size));
      const int kMaxFeatureSize = 4096;
      CHECK_GE(feature_size, 0);
      CHECK_LE(feature_size, kMaxFeatureSize);
      auto batch_size = c->Dim(c->input(0), 0);
      c->set_output(0, c->MakeShape({batch_size, InferenceContext::kUnknownDim,
                                     feature_size}));
      c->set_output(1, c->MakeShape({1}));
      c->set_output(2, c->MakeShape({batch_size}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
This op referred to as Ternary Sequence String Projection op (TSP), tokenizes
input text either on space or unicode boundary. Fingerprint for each token is
computed using murmur hash and bit features are extracted from the fingerprint
that maps every 2 bits to the ternary output {-1, 0, 1}. This effectively turns
a batch of text input into a ternary rank 3 tensor (in float format) of shape
[batch size, max token length, requested number of features].

Input(s):
- input: A string tensor with batch size number of elements.

Attribute(s):
- feature_size: Length of the ternary vector generated for each token.
- distortion_probability: When non zero distort the input text with this
    probability. Helps as a regularization method when training data set is
    small.
- vocabulary: When not empty provides a list of unique unicode characters that
    will be allowed in the input text before fingerprinting. Another way to
    say it is that the vocabulary is an optional character allowlist for the
    input text. It helps normalize the text.
- hashtype: Hashing method to use for projection.
- max_splits: Maximum number of tokens that are allowed. It helps restrict the
    max token length of the projection output. When the value is -1 the op
    does not restrict the number of tokens in the output.
- exclude_nonalphaspace_unicodes: When true excludes all unicodes that are
    not alphabets or space character. This is multilingual. Though the effect
    of this flag can be achieved using vocabulary, the vocabulary will have to
    be very large for multilingual input.
- add_bos_tag: When true inserts a begin of sentence tag.
- add_eos_tag: When true inserts a end of sentence tag.
- word_novelty_bits: When true adds a special feature to the ternary output
    that captures the frequency of occurrence of a particular token. This is an
    experimental feature.
- doc_size_levels: When true adds a special feature to the ternary projection
    output the document size in log scale. This is an experimental feature.
- split_on_space: When true tokenization is done on space segmentation.
    Otherwise tokenization is done by segmenting on unicode boundary.
- add_first_cap_feature: Specifies the probability with which a feature to the
     resulting projection tensor that helps discriminate if the input token is
     Camel case will be added.
- add_all_caps_feature: Specifies the probability with which a feature to the
    resulting projection tensor that helps discriminate if the input token is
    ALLCAPS will be added.

Output(s):
- projection: Floating point tensor with ternary values of shape
    [batch size, max token length, requested number of features].
- dummy_output: Ignore this output, will be eliminated in a subsequent version.
- sequence_length: Batch size length vector containing the number of tokens for
    each input text entry.
)doc");
