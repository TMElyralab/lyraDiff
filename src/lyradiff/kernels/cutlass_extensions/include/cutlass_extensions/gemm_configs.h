

#pragma once

namespace lyradiff {

namespace cutlass_extensions {
// Note: The shapes are in the format MxNxK. The K shape of the runtime config MUST match the K shape
//       in the kernel layout details when doing weight only quantization.
enum class CutlassTileConfig {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // SiMT config
    CtaShape128x128x8_WarpShape64x64x8,

    // TensorCore configs CTA_N = 128, CTA_K = 64
    // Warp configs for M=16
    CtaShape16x128x64_WarpShape16x32x64,
    // Warp configs for M=32
    CtaShape32x128x64_WarpShape32x32x64,

    // Warp configs for M=64
    CtaShape64x128x64_WarpShape32x64x64,
    CtaShape64x64x128_WarpShape32x64x64,
    CtaShape64x128x64_WarpShape64x32x64,
    CtaShape64x128x64_WarpShape64x64x64,

    // Warp configs for M=128
    CtaShape128x64x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x32x64,
    CtaShape128x128x64_WarpShape64x64x64,
    CtaShape128x128x64_WarpShape128x32x64,
    CtaShape128x256x64_WarpShape64x64x64,

    // Warp configs for M=256
    CtaShape256x128x64_WarpShape64x64x64,

    // TensorCore config CTA_N = 256, CTA_K = 64
    CtaShape16x256x64_WarpShape16x64x64,

    // New Added config for lyradiff
    CtaShape128x128x64_WarpShape64x64x32,
    // CtaShape128x128x64_WarpShape64x32x32,
    CtaShape128x64x64_WarpShape64x64x64,
    CtaShape128x64x64_WarpShape64x64x32,
    // CtaShape128x64x64_WarpShape64x32x32,
    CtaShape128x64x32_WarpShape64x64x32,
    CtaShape128x64x32_WarpShape64x32x32,
};

enum class SplitKStyle {
    NO_SPLIT_K,
    SPLIT_K_SERIAL,
    // SPLIT_K_PARALLEL // Not supported yet
};

enum class CutlassTileConfigSM90 {
    // Signals that we should run heuristics do choose a config
    Undefined,

    // Signals that we should run heuristics do choose a config
    ChooseWithHeuristic,

    // CTA configs for M=64
    CtaShape64x16x128B,
    CtaShape64x32x128B,
    CtaShape64x64x128B,
    CtaShape64x128x128B,
    CtaShape64x256x128B,

    // CTA configs for M=128
    CtaShape128x16x128B,
    CtaShape128x32x128B,
    CtaShape128x64x128B,
    CtaShape128x128x128B,
    CtaShape128x256x128B,
};

enum class MainloopScheduleType {
    AUTO  // Automatically selects between pingpong and cooperative schedules on Hopper. On older architectures, this
          // defaults to the "legacy" main loop schedule.
};

enum class EpilogueScheduleType {
    AUTO  // Automatically chooses an epilogue schedule compatible with the selected main loop schedule for Hopper. For
          // architectures older than hopper, the epilogue is always performed by the same thread block as the main
          // loop.
};

enum class ClusterShape {
    ClusterShape_1x1x1,
    ClusterShape_2x1x1,
    ClusterShape_1x2x1,
    ClusterShape_2x2x1
};

struct CutlassGemmConfig {
    CutlassTileConfig tile_config    = CutlassTileConfig::ChooseWithHeuristic;
    SplitKStyle       split_k_style  = SplitKStyle::NO_SPLIT_K;
    int               split_k_factor = -1;
    int               stages         = -1;

    // config options for sm90
    CutlassTileConfigSM90 tile_config_sm90  = CutlassTileConfigSM90::ChooseWithHeuristic;
    MainloopScheduleType  mainloop_schedule = MainloopScheduleType::AUTO;
    EpilogueScheduleType  epilogue_schedule = EpilogueScheduleType::AUTO;
    ClusterShape          cluster_shape     = ClusterShape::ClusterShape_1x1x1;

    CutlassGemmConfig() {}

    CutlassGemmConfig(CutlassTileConfig tile_config, SplitKStyle split_k_style, int split_k_factor, int stages):
        tile_config(tile_config), split_k_style(split_k_style), split_k_factor(split_k_factor), stages(stages)
    {
    }

    CutlassGemmConfig(CutlassTileConfigSM90 tile_config_sm90,
                      MainloopScheduleType  mainloop_schedule,
                      EpilogueScheduleType  epilogue_schedule,
                      ClusterShape          cluster_shape):
        tile_config_sm90(tile_config_sm90),
        mainloop_schedule(mainloop_schedule),
        epilogue_schedule(epilogue_schedule),
        cluster_shape(cluster_shape)
    {
    }
};

}  // namespace cutlass_extensions
}  // namespace lyradiff
