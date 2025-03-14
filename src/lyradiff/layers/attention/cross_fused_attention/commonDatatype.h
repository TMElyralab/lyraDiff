#pragma once
#include <stdint.h>

namespace lyradiff {

namespace cross_attn {

// Do not modify this, it is integrated from src/fused_multihead_attention_utils.h in fmha_v2.
enum MHCADataType {
    DATA_TYPE_BOOL,
    DATA_TYPE_E8M10,
    DATA_TYPE_E8M7,
    DATA_TYPE_FP16,
    DATA_TYPE_FP32,
    DATA_TYPE_INT4,
    DATA_TYPE_INT8,
    DATA_TYPE_INT32
};

#define ENABLE_SM75
#define ENABLE_SM80
#define ENABLE_SM86
#define ENABLE_SM89

constexpr uint32_t kSM_75 = 75;
constexpr uint32_t kSM_80 = 80;
constexpr uint32_t kSM_86 = 86;
constexpr uint32_t kSM_89 = 89;

}  // namespace cross_attn
}  // namespace lyradiff
