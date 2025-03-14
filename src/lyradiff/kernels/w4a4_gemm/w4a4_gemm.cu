#include "src/lyradiff/reduce.h"
#include "src/lyradiff/utils/cuda_type_utils.cuh"
#include "src/lyradiff/utils/cuda_utils.h"
#include "w4a4_gemm.h"
#include <cute/tensor.hpp>

namespace lyradiff {
using namespace cute;

namespace config {

template<int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 128, int kStage_ = 2, int kSmemLayoutCBatch_ = 2>
struct GemmConfig {
    using T_IN      = cute::int4b_t;
    using T_OUT     = cute::bfloat16_t;
    using T_COMPUTE = float;
    using T_SCALE   = float;
    using T_LORA    = cute::bfloat16_t;

    // tile configuration
    static constexpr int kTileM            = kTileM_;
    static constexpr int kTileN            = kTileN_;
    static constexpr int kTileK            = kTileK_;
    static constexpr int kLoraTileK        = 32;
    static constexpr int kStage            = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    static constexpr int kTileMK = kTileM * kTileK;
    static constexpr int kTileNK = kTileN * kTileK;

    static constexpr int ScaleGroupDim = 64;

    using SmemLayoutAtom = decltype(composition(
        Swizzle<2, 5, 5>{}, make_layout(make_shape(Int<4>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}))));
    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtom{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    // using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    // using mma_op = SM80_16x8x8_F32TF32TF32F32_TN;
    using mma_op = SM80_16x8x64_S32S4S4S32_TN;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape        = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT =
        decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    static constexpr int KRepeat     = kTileK / kMmaPK;
    static constexpr int kTileScaleM = kTileM * KRepeat;
    static constexpr int kTileScaleN = kTileN * KRepeat;

    using SmemLayoutScaleA = decltype(make_layout(make_shape(Int<kTileM>{}, Int<KRepeat>{}, Int<kStage>{}),
                                                  make_stride(Int<KRepeat>{}, Int<1>{}, Int<kTileScaleM>{})));

    using SmemLayoutScaleB = decltype(make_layout(make_shape(Int<kTileN>{}, Int<KRepeat>{}, Int<kStage>{}),
                                                  make_stride(Int<KRepeat>{}, Int<1>{}, Int<kTileScaleN>{})));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T_IN>;

    // using g2s_scale_copy_atom = Copy_Atom<g2s_copy_traits, T_SCALE>;

    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<32>{}))));
    using G2SCopyB = G2SCopyA;

    // using G2ScaleCopyA =
    //     decltype(make_tiled_copy(g2s_scale_copy_atom{},
    //                              make_layout(make_shape(Int<KRepeat>{}, Int<32>{}), make_stride(Int<32>{},
    //                              Int<1>{})), make_layout(make_shape(Int<1>{}, Int<4>{}))));
    // using G2ScaleCopyB = G2ScaleCopyA;

    // shared memory to register copy
    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T_IN>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // epilogue: register to global via shared memory
    using SmemLayoutAtomC = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}), make_stride(Int<kMmaPN>{}, Int<1>{}))));
    using SmemLayoutC =
        decltype(tile_to_shape(SmemLayoutAtomC{}, make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemLayoutCBatch>{})));

    static_assert(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) >= size(SmemLayoutC{}),
                  "C shared memory request is large than A's one pipe");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T_OUT>;

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T_OUT>;
    using S2GCopyC =
        decltype(make_tiled_copy(S2GCopyAtomC{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    static constexpr int kThreadNum     = size(MMA{});
    static constexpr int shm_size_AB    = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    static constexpr int shm_size_C     = cute::cosize(SmemLayoutC{});
    static constexpr int shm_size_SCALE = cute::cosize(SmemLayoutScaleA{}) + cute::cosize(SmemLayoutScaleB{});

    static constexpr int kShmSize = shm_size_AB * sizeof(T_IN);
        // cute::max(shm_size_AB * sizeof(T_IN) + shm_size_SCALE * sizeof(T_SCALE), shm_size_C * sizeof(T_OUT));
};

template<int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32, int kStage_ = 2, int kSmemLayoutCBatch_ = 2>
struct LoraUpGemmConfig {
    using T_IN      = cute::bfloat16_t;
    using T_COMPUTE = float;
    // using T_OUT     = float;

    // tile configuration
    static constexpr int kTileM            = kTileM_;
    static constexpr int kTileN            = kTileN_;
    static constexpr int kTileK            = kTileK_;
    static constexpr int kStage            = kStage_;
    static constexpr int kSmemLayoutCBatch = kSmemLayoutCBatch_;

    static constexpr int kTileMK = kTileM * kTileK;
    static constexpr int kTileNK = kTileN * kTileK;

    static constexpr int ScaleGroupDim = 64;

    using SmemLayoutAtomA = decltype(composition(
        Swizzle<2, 3, 3>{}, make_layout(make_shape(Int<8>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}))));
    // using SmemLayoutAtomB = decltype(composition(
    //     Swizzle<2, 2, 2>{}, make_layout(make_shape(Int<8>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}))));

    using SmemLayoutA =
        decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutB =
        decltype(tile_to_shape(SmemLayoutAtomA{}, make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;

    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape        = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT =
        decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using g2s_copy_op     = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom   = Copy_Atom<g2s_copy_traits, T_IN>;
    // using g2s_copy_atom_b = Copy_Atom<g2s_copy_traits, T_IN>;

    using G2SCopyA =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}))));

    using G2SCopyB = G2SCopyA;
    // using G2SCopyB =
    //     decltype(make_tiled_copy(g2s_copy_atom_b{},
    //                              make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
    //                              make_layout(make_shape(Int<1>{}, Int<8>{}))));

    // shared memory to register copy
    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T_IN>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    static constexpr int kThreadNum  = size(MMA{});
    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{});
    //

    static constexpr int kShmSize = shm_size_AB * sizeof(T_IN);
};

}  // namespace config

__device__ __forceinline__ uint32_t quantize_float2(float2 value)
{
    int      v1, v2;
    uint32_t result;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v1) : "f"(value.x));
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(v2) : "f"(value.y));
    asm volatile("cvt.pack.sat.s4.s32.b32 %0, %1, %2, 0;" : "=r"(result) : "r"(v2), "r"(v1));
    return result;
}

__forceinline__ __device__ void _atomic_addh2(half2* addr, half2 in)
{
    int in_int = *((int*)&in);
    asm("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;" ::"l"(addr), "r"(in_int));
}

__device__ __forceinline__ static void reduce_add(float* addr, float val)
{
    asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;" ::"l"(addr), "f"(val));
}

template<typename Config>
CUTE_DEVICE auto w4a4_gemm(void*        Dptr,
                           const void*  Aptr,
                           const void*  Bptr,
                           const float* AweightScale,
                           const float* BweightScale,
                           int          m,
                           int          n,
                           int          k,
                           int          groups,
                           void*        shm_ptr)
{
    using namespace cute;
    using X = Underscore;

    using T_IN  = typename Config::T_IN;       // int4
    using T_OUT = typename Config::T_COMPUTE;  // fp32
    // using T_COMPUTE   = typename Config::T_COMPUTE;
    // using T_LORA      = typename Config::T_LORA;
    using T_SCALE     = typename Config::T_SCALE;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    // using SmemLayoutScaleA = typename Config::SmemLayoutScaleA;
    // using SmemLayoutScaleB = typename Config::SmemLayoutScaleB;
    // using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA     = typename Config::G2SCopyA;
    using G2SCopyB     = typename Config::G2SCopyB;
    // using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    // using S2GCopyAtomC = typename Config::S2GCopyAtomC;

    constexpr int kTileM  = Config::kTileM;
    constexpr int kTileN  = Config::kTileN;
    constexpr int kTileK  = Config::kTileK;
    constexpr int kStage  = Config::kStage;
    constexpr int KRepeat = Config::KRepeat;

    T_IN* shm_data = (T_IN*)shm_ptr;

    T_IN* Ashm = shm_data;
    T_IN* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    // T_SCALE* AScaleShm = reinterpret_cast<T_SCALE*>(Bshm + cute::cosize(SmemLayoutB{}));
    // T_SCALE* BScaleShm = AScaleShm + cute::cosize(SmemLayoutScaleA{});

    int idx = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    int m_scale_idx[10];
    int n_scale_idx[10];

    // use Tensor notation to represent device pointer + dimension
    cute::Tensor A = make_tensor(make_gmem_ptr<T_IN>(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));   // (M, K)
    cute::Tensor B = make_tensor(make_gmem_ptr<T_IN>(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));   // (N, K)
    cute::Tensor D = make_tensor(make_gmem_ptr<T_OUT>(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)

    // slice the tensor to small one which is used for current thread block.
    cute::Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k)
    cute::Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, k)
    cute::Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));  // (kTileM, kTileN)

    // if (idx == 0 && ix == 0 && iy == 0) {
    //     cute::print_tensor(gB(_, _, 0));
    // }

    cute::Tensor Ascale =
        make_tensor(make_gmem_ptr(AweightScale), make_shape(m, groups), make_stride(groups, Int<1>{}));
    cute::Tensor Bscale =
        make_tensor(make_gmem_ptr(BweightScale), make_shape(n, groups), make_stride(groups, Int<1>{}));

    cute::Tensor gAscale =
        local_tile(Ascale, make_tile(Int<kTileM>{}, Int<1>{}), make_coord(iy, _));  // (kTileN, 1, group)
    cute::Tensor gBscale =
        local_tile(Bscale, make_tile(Int<kTileN>{}, Int<1>{}), make_coord(ix, _));  // (kTileN, 1, group)

    // cute::Tensor gAscaleS =
    //     local_tile(Ascale, make_tile(Int<kTileM>{}, Int<KRepeat>{}), make_coord(iy, _));  // (kTileN, 2, itile)
    // cute::Tensor gBscaleS =
    //     local_tile(Bscale, make_tile(Int<kTileN>{}, Int<KRepeat>{}), make_coord(ix, _));  // (kTileN, 2, itile)

    // auto rAscale = make_tensor<float>(make_shape(Int<kTileM>{}, Int<1>{}), make_stride(Int<1>{}, Int<0>{}));
    // auto rBscale = make_tensor<float>(make_shape(Int<kTileN>{}, Int<1>{}), make_stride(Int<1>{}, Int<0>{}));

    // shared memory
    auto sA = make_tensor(make_smem_ptr<T_IN>(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr<T_IN>(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage)

    // auto sScaleA = make_tensor(make_smem_ptr<T_SCALE>(AScaleShm), SmemLayoutScaleA{});  // (kTileM, kTileK, kStage)
    // auto sScaleB = make_tensor(make_smem_ptr<T_SCALE>(BScaleShm), SmemLayoutScaleB{});  // (kTileN, kTileK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto     thr_mma = tiled_mma.get_slice(idx);
    auto     tCrA    = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto     tCrB    = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto     tCrD    = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

    auto tCgC = thr_mma.partition_C(gD);  // (MMA, MMA_M, MMA_N)

    auto res_buf = make_tensor<T_OUT>(tCrD.layout());    // (MMA, MMA_M, MMA_N)
    auto zeros   = make_tensor<int32_t>(tCrD.layout());  // (MMA, MMA_M, MMA_N)

    // fill zero for accumulator
    // clear(tCrD);
    clear(zeros);
    clear(res_buf);

    // To calculate offset for scale
    int m_offset = iy * kTileM;
    int n_offset = ix * kTileN;

    int num_tile_k = size<2>(gA);
    int num_tile_m = size<1>(tCrD);
    int num_tile_n = size<2>(tCrD);

    // int tid      = threadIdx.x;
    int warp_id  = threadIdx.x / 32;
    int warp_idx = threadIdx.x % 32;

    int warp_m_idx = warp_id % 2;
    int warp_n_idx = warp_id / 2;

    // TODO: the idx calc only support our current mma setting, fix later @yibolu
    using AtomShape_MNK       = typename TiledMMA::AtomShape_MNK;
    constexpr int mma_m_size  = size<0>(AtomShape_MNK{});
    constexpr int mma_n_size  = size<1>(AtomShape_MNK{});
    constexpr int tile_m_size = size<0>(AtomShape_MNK{}) * 2;
    constexpr int tile_n_size = size<1>(AtomShape_MNK{}) * 2;

    m_offset = warp_m_idx * mma_m_size;
    n_offset = warp_n_idx * mma_n_size;

    for (int m_idx = 0; m_idx < num_tile_m; m_idx++) {
        // int cur_m          = m_offset + m_idx * tile_m_size + warp_idx / 4;
        m_scale_idx[m_idx] = m_offset + m_idx * tile_m_size + warp_idx / 4;
    }

    for (int n_idx = 0; n_idx < num_tile_n; n_idx++) {
        // int cur_n          = n_offset + n_idx * tile_n_size + (warp_idx % 4) * 2;
        n_scale_idx[n_idx] = n_offset + n_idx * tile_n_size + (warp_idx % 4) * 2;
    }

    // gmem -cp.async-> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);   // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);   // ? (CPY, CPY_M, CPY_K)

    G2SCopyA g2s_tiled_copy_a;
    auto     g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto     tAgA_copy      = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto     tAsA_copy      = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto     g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto     tBgB_copy      = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto     tBsB_copy      = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

    // submit kStage - 1 tile
    // gmem -> shm
#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();

        // auto cur_s_scale_a = sScaleA(_, _, istage);
        // auto cur_s_scale_b = sScaleB(_, _, istage);
        // auto cur_g_scale_a = gAscaleS(_, _, istage);
        // auto cur_g_scale_b = gBscaleS(_, _, istage);

        // cur_s_scale_a(idx * 2)     = cur_g_scale_a(idx * 2);
        // cur_s_scale_a(idx * 2 + 1) = cur_g_scale_a(idx * 2 + 1);
        // cur_s_scale_b(idx * 2)     = cur_g_scale_b(idx * 2);
        // cur_s_scale_b(idx * 2 + 1) = cur_g_scale_b(idx * 2 + 1);

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / kTileK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kStage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

                    // auto cur_s_scale_a = sScaleA(_, _, ismem_write);
                    // auto cur_s_scale_b = sScaleB(_, _, ismem_write);
                    // auto cur_g_scale_a = gAscaleS(_, _, itile_to_read);
                    // auto cur_g_scale_b = gBscaleS(_, _, itile_to_read);

                    // cur_s_scale_a(idx * 2)     = cur_g_scale_a(idx * 2);
                    // cur_s_scale_a(idx * 2 + 1) = cur_g_scale_a(idx * 2 + 1);
                    // cur_s_scale_b(idx * 2)     = cur_g_scale_b(idx * 2);
                    // cur_s_scale_b(idx * 2 + 1) = cur_g_scale_b(idx * 2 + 1);

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), zeros);
            auto rAscale = gAscale(_, _, itile * nk + ik);
            auto rBscale = gBscale(_, _, itile * nk + ik);

            // auto cur_a_scale = sScaleA(_, ik, ismem_read);
            // auto cur_b_scale = sScaleB(_, ik, ismem_read);

#pragma unroll
            for (int m_idx = 0; m_idx < num_tile_m; m_idx++) {
                int    cur_m = m_scale_idx[m_idx];
                float2 m_v   = make_float2(rAscale(cur_m), rAscale(cur_m + 8));

                for (int n_idx = 0; n_idx < num_tile_n; n_idx++) {
                    // int cur_m = m_offset + m_idx * tile_m_size + warp_idx / 4;
                    // int cur_n = n_offset + n_idx * tile_n_size + (warp_idx % 4) * 2;

                    int cur_n = n_scale_idx[n_idx];

                    auto cur_src_tensor = tCrD(_, m_idx, n_idx);
                    auto cur_dst_tensor = res_buf(_, m_idx, n_idx);

                    float2 n_v = make_float2(rBscale(cur_n), rBscale(cur_n + 1));

                    // float2 m_v = make_float2(rAscale(0), rAscale(1));
                    // float2 n_v = make_float2(rBscale(2), rBscale(3));

                    // float2 m_v = make_float2(cuda_cast<float>(1.0f), cuda_cast<float>(1.0f));
                    // float2 n_v = make_float2(cuda_cast<float>(1.0f), cuda_cast<float>(1.0f));

                    cur_dst_tensor(0) += __int2float_rn(cur_src_tensor(0)) * m_v.x * n_v.x;
                    cur_dst_tensor(1) += __int2float_rn(cur_src_tensor(1)) * m_v.x * n_v.y;
                    cur_dst_tensor(2) += __int2float_rn(cur_src_tensor(2)) * m_v.y * n_v.x;
                    cur_dst_tensor(3) += __int2float_rn(cur_src_tensor(3)) * m_v.y * n_v.y;
                }
            }
        }  // for ik
    }  // itile
    // cute::copy(tCrD, res_buf);
    return res_buf;
}

template<typename Config>
CUTE_DEVICE auto lora_up_gemm(
    void* Dptr, void* res_ptr, const void* Aptr, const void* Bptr, int m, int n, int k, int groups, void* shm_ptr)
{
    using namespace cute;
    using X = Underscore;

    using T_IN  = typename Config::T_IN;
    using T_OUT = typename Config::T_COMPUTE;
    // using T_COMPUTE   = typename Config::T_COMPUTE;
    // using T_LORA      = typename Config::T_LORA;
    // using T_SCALE     = typename Config::T_SCALE;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    // using SmemLayoutC = typename Config::SmemLayoutC;
    using TiledMMA = typename Config::MMA;

    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using G2SCopyA     = typename Config::G2SCopyA;
    using G2SCopyB     = typename Config::G2SCopyB;
    // using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    // using S2GCopyAtomC = typename Config::S2GCopyAtomC;
    // using S2GCopyC     = typename Config::S2GCopyC;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    T_IN* shm_data = (T_IN*)shm_ptr;

    T_IN* Ashm = shm_data;
    T_IN* Bshm = shm_data + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    // use Tensor notation to represent device pointer + dimension
    cute::Tensor A = make_tensor(make_gmem_ptr<T_IN>(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));   // (M, K)
    cute::Tensor B = make_tensor(make_gmem_ptr<T_IN>(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));   // (N, K)
    cute::Tensor D = make_tensor(make_gmem_ptr<T_OUT>(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)

    // slice the tensor to small one which is used for current thread block.
    cute::Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));  // (kTileM, kTileK, k)
    cute::Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));  // (kTileN, kTileK, k)
    cute::Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));  // (kTileM, kTileN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr<T_IN>(Ashm), SmemLayoutA{});  // (kTileM, kTileK, kStage)
    auto sB = make_tensor(make_smem_ptr<T_IN>(Bshm), SmemLayoutB{});  // (kTileN, kTileK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto     thr_mma = tiled_mma.get_slice(idx);
    auto     tCrA    = thr_mma.partition_fragment_A(gA(_, _, 0));  // (MMA, MMA_M, MMA_K)
    auto     tCrB    = thr_mma.partition_fragment_B(gB(_, _, 0));  // (MMA, MMA_N, MMA_K)
    auto     tCrD    = thr_mma.partition_fragment_C(gD);           // (MMA, MMA_M, MMA_N)

    // clear(tCrD);

    auto res = make_tensor(make_rmem_ptr<float>(res_ptr), tCrD.layout());

    // gmem -cp.async-> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tCrA);   // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tCrB);   // ? (CPY, CPY_M, CPY_K)

    G2SCopyA g2s_tiled_copy_a;
    auto     g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto     tAgA_copy      = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto     tAsA_copy      = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    G2SCopyB g2s_tiled_copy_b;
    auto     g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto     tBgB_copy      = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto     tBsB_copy      = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    int itile_to_read = 0;
    int ismem_read    = 0;
    int ismem_write   = 0;

    // submit kStage - 1 tile
    // gmem -> shm
#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / kTileK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kStage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));

                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, res, tCrA(_, _, ik), tCrB(_, _, ik), res);
        }  // for ik
    }  // itile

    return res;
}

template<typename Config, typename LoraConfig>
__global__ void fused_w4a4_gemm_and_lora_up_multistage_v2(void*        Dptr,
                                                          const void*  Aptr,
                                                          const void*  Bptr,
                                                          const void*  LoraAptr,
                                                          const void*  LoraBptr,
                                                          const float* AweightScale,
                                                          const float* BweightScale,
                                                          int          m,
                                                          int          n,
                                                          int          k,
                                                          int          lora_k,
                                                          int          groups)
{
    using namespace cute;
    using X = Underscore;

    using T_IN      = typename Config::T_IN;
    using T_OUT     = typename Config::T_OUT;
    using T_COMPUTE = typename Config::T_COMPUTE;
    using TiledMMA  = typename Config::MMA;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;
    constexpr int kStage = Config::kStage;

    extern __shared__ T_IN shm_data[];

    int idx = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    cute::Tensor D = make_tensor(make_gmem_ptr<T_OUT>(Dptr), make_shape(m, n), make_stride(n, Int<1>{}));  // (M, N)

    cute::Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));  // (kTileM, kTileN)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    // method
    TiledMMA tiled_mma;
    auto     thr_mma = tiled_mma.get_slice(idx);
    auto     tCrD    = thr_mma.partition_fragment_C(gD);  // (MMA, MMA_M, MMA_N)

    auto tCgC = thr_mma.partition_C(gD);  // (MMA, MMA_M, MMA_N)

    auto res_buf = w4a4_gemm<Config>(Dptr, Aptr, Bptr, AweightScale, BweightScale, m, n, k, groups, shm_data);

    auto lora_res_buf =
        lora_up_gemm<LoraConfig>(Dptr, res_buf.data(), LoraAptr, LoraBptr, m, n, lora_k, groups, shm_data);

    // axpby(1.0f, lora_res_buf, 1.0f, res_buf);

    cute::copy(res_buf, tCgC);

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    //     auto sC = make_tensor(make_smem_ptr<T_OUT>(Ashm), SmemLayoutC{});

    //     auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
    //     auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
    //     auto tCrC_r2s         = r2s_thr_copy_c.retile_S(tCrD);   // (CPY, CPY_M, CPY_N)
    //     auto tCsC_r2s         = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

    //     if (idx == 0 && ix == 0 && iy == 0) {
    //         // printf("nk: %d, ntile: %d \n", nk, k / kTileK);
    //         cute::print(tCrD);
    //         cute::print(tCrC_r2s);
    //         cute::print(tCsC_r2s);
    //         // cute::print(tAgA_copy);
    //     }

    //     S2GCopyC s2g_tiled_copy_c;
    //     auto     s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    //     auto     tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);  // (CPY, _1, _1, pipe)
    //     auto     tCgC_s2g       = s2g_thr_copy_c.partition_D(gD);  // (CPY, CPY_M, CPY_N)

    //     auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);  // (CPY_, CPY_MN)
    //     auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);  // (CPY_, CPY_MN)

    //     int step = size<3>(tCsC_r2s);  // pipe
    // #pragma unroll
    //     for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
    //         // reg -> shm
    // #pragma unroll
    //         for (int j = 0; j < step; ++j) {
    //             // we add a temp tensor to cope with accumulator and output data type
    //             // difference
    //             auto t = make_tensor_like<T_OUT>(tCrC_r2sx(_, i + j));
    //             cute::copy(tCrC_r2sx(_, i + j), t);

    //             cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
    //         }
    //         __syncthreads();

    // #pragma unroll
    //         // shm -> global
    //         for (int j = 0; j < step; ++j) {
    //             cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
    //         }

    //         __syncthreads();
    //     }
}

template<int kTileM, int kTileN, int kTileK, typename TiledMMA>
__global__ void fused_quantized_and_lora_down_simple(float*       Cptr,
                                                     uint32_t*    quantized_result,
                                                     float*       weight_scale,
                                                     const void*  Aptr,
                                                     const void*  Bptr,
                                                     const float* Smoothptr,
                                                     int          m,
                                                     int          n,
                                                     int          k)
{
    using T_IN = cute::bfloat16_t;

    using namespace cute;

    auto SmemLayoutA      = make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}));
    auto SmemLayoutB      = make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}));
    auto SmemLayoutSmooth = make_layout(make_shape(Int<kTileK>{}), make_stride(Int<1>{}));

    TiledCopy g2s_tiled_copy_a =
        make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, T_IN>{},
                        make_layout(make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{})),
                        make_layout(make_shape(Int<1>{}, Int<8>{})));

    TiledCopy g2s_tiled_copy_b = g2s_tiled_copy_a;

    TiledCopy g2s_tiled_copy_smooth = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, float>{},
                                                      make_layout(make_shape(Int<8>{}), make_stride(Int<1>{})),
                                                      make_layout(Int<4>{}));

    using s2r_copy_op     = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom   = Copy_Atom<s2r_copy_traits, T_IN>;

    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    static constexpr int WARP_SIZE = 32;

    static constexpr int shm_size_AB     = cute::cosize(SmemLayoutA) + cute::cosize(SmemLayoutB);
    static constexpr int shm_size_Smooth = cute::cosize(SmemLayoutSmooth);

    __shared__ T_IN  shm_data[shm_size_AB];
    __shared__ float shm_data_smooth[shm_size_Smooth];

    float* Sshm = shm_data_smooth;

    T_IN* Ashm = shm_data;
    T_IN* Bshm = shm_data + cute::cosize(SmemLayoutA);

    cute::Tensor A     = make_tensor(make_gmem_ptr<T_IN>(Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
    cute::Tensor B     = make_tensor(make_gmem_ptr<T_IN>(Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
    cute::Tensor C     = make_tensor(make_gmem_ptr(Cptr), make_shape(m, n), make_stride(n, Int<1>{}));
    cute::Tensor Scale = make_tensor(make_gmem_ptr(weight_scale), make_shape(m, k / 64), make_stride(k / 64, Int<1>{}));
    cute::Tensor Smooth = make_tensor(make_gmem_ptr(Smoothptr), make_shape(k), make_stride(Int<1>{}));
    cute::Tensor QuantizedResult =
        make_tensor(make_gmem_ptr(quantized_result), make_shape(m, k / 8), make_stride(k / 8, Int<1>{}));
    cute::Tensor QuantizedResultInt4 =
        make_tensor(make_gmem_ptr<cute::int4_t>(quantized_result), make_shape(m, k), make_stride(k, Int<1>{}));

    int tid = threadIdx.x;
    int ix  = blockIdx.x;
    int iy  = blockIdx.y;

    cute::Tensor gA      = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, ix));
    cute::Tensor gB      = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(Int<0>{}, ix));
    cute::Tensor gC      = local_tile(C, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, Int<0>{}));
    cute::Tensor gScale  = local_tile(Scale, make_tile(Int<kTileM>{}, Int<1>{}), make_coord(iy, ix));
    cute::Tensor gSmooth = local_tile(Smooth, make_tile(Int<kTileK>{}), make_coord(ix));
    cute::Tensor gQuantizedResult =
        local_tile(QuantizedResult, make_tile(Int<kTileM>{}, Int<kTileK / 8>{}), make_coord(iy, ix));

    auto sA      = make_tensor(make_smem_ptr<T_IN>(Ashm), SmemLayoutA);  // (kTileM, kTileK, kStage)
    auto sB      = make_tensor(make_smem_ptr<T_IN>(Bshm), SmemLayoutB);  // (kTileN, kTileK, kStage)
    auto sSmooth = make_tensor(make_smem_ptr<float>(Sshm), SmemLayoutSmooth);

    TiledMMA tiled_mma;
    auto     thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto     tAgA    = thr_mma.partition_A(gA);  // (MMA, MMA_M, MMA_K, num_tile_k)
    auto     tBgB    = thr_mma.partition_B(gB);  // (MMA, MMA_N, MMA_K, num_tile_k)
    auto     tCgC    = thr_mma.partition_C(gC);  // (MMA, MMA_M, MMA_N)

    // clear(tCgC);

    auto tArA = thr_mma.partition_fragment_A(gA);  // (MMA, MMA_M, MMA_K)
    auto tBrB = thr_mma.partition_fragment_B(gB);  // (MMA, MMA_N, MMA_K)
    auto tCrC = thr_mma.partition_fragment_C(gC);  // (MMA, MMA_M, MMA_N)

    clear(tCrC);
    // gmem --> shm -ldmatrix-> reg
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(tid);
    auto tAsA             = s2r_thr_copy_a.partition_S(sA);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrA_view        = s2r_thr_copy_a.retile_D(tArA);   // ? (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(tid);
    auto tBsB             = s2r_thr_copy_b.partition_S(sB);  // ? (CPY, CPY_M, CPY_K, kStage)
    auto tCrB_view        = s2r_thr_copy_b.retile_D(tBrB);   // ? (CPY, CPY_M, CPY_K)

    // G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(tid);
    auto tAgA_copy      = g2s_thr_copy_a.partition_S(gA);  // (CPY, CPY_M, CPY_K, k)
    auto tAsA_copy      = g2s_thr_copy_a.partition_D(sA);  // (CPY, CPY_M, CPY_K, kStage)

    // G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(tid);
    auto tBgB_copy      = g2s_thr_copy_b.partition_S(gB);  // (CPY, CPY_N, CPY_K, k)
    auto tBsB_copy      = g2s_thr_copy_b.partition_D(sB);  // (CPY, CPY_N, CPY_K, kStage)

    // auto g2s_thr_copy_smooth = g2s_tiled_copy_smooth.get_slice(tid);
    // auto tSgS_copy           = g2s_thr_copy_smooth.partition_S(gSmooth);  // (CPY, CPY_M, CPY_K, k)
    // auto tSsS_copy           = g2s_thr_copy_smooth.partition_D(sSmooth);  // (CPY, CPY_M, CPY_K, kStage)

    cute::copy(g2s_tiled_copy_a, tAgA_copy, tAsA_copy);
    cute::copy(g2s_tiled_copy_b, tBgB_copy, tBsB_copy);
    if (tid < kTileK) {  // g2s_thr_copy_smooth 一直失败，暂时用这个办法
        sSmooth(tid) = gSmooth(tid);
    }

    // 在这__syncthreads 一下，等待所有的拷贝完成
    __syncthreads();

    // 这里需要 quantize input
    // share smem sA 的 shape 为 tile_m, tile_k。这里 tile_k 被设置为 64，刚好和 groupsize 一致
    // 这里我们因为 MMA 的设置，线程数定死为 128，总共 4 个 warp，我们设置每个 warp 计算一行的 最大值 以及 quantize 保存
    //

    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    static constexpr int repeat_m = kTileM / 4;
    for (int i = 0; i < repeat_m; i++) {
        auto   cur_line   = sA(i * 4 + warp_id, _);
        int    cur_idx    = lane_id * 2;
        float2 cur_v      = make_float2(cuda_cast<float>(cur_line(cur_idx)), cuda_cast<float>(cur_line(cur_idx + 1)));
        float2 cur_smooth = make_float2(sSmooth(cur_idx), sSmooth(cur_idx + 1));
        cur_v             = cur_v / cur_smooth;
        float2 cur_amax   = fabs(cur_v);
        float  cur_max    = max(cur_amax.x, cur_amax.y);
        float  cur_scale  = warpReduceMax(cur_max) / 7.0f;

        cur_v.x = cur_v.x / cur_scale;
        cur_v.y = cur_v.y / cur_scale;

        uint32_t quantized_result = quantize_float2(cur_v) << (lane_id % 4 * 8);

#pragma unroll
        for (int mask = 1; mask <= 2; mask *= 2) {
            quantized_result |= __shfl_xor_sync(~0, quantized_result, mask);
        }

        if (lane_id == 0) {
            gScale(i * 4 + warp_id) = cur_scale;
        }

        auto cur_quantized_result = gQuantizedResult(i * 4 + warp_id, _);

        if (lane_id % 4 == 0) {
            cur_quantized_result(lane_id / 4) = quantized_result;
        }
    }

    // 后续完成 gemm
    cute::copy(s2r_tiled_copy_a, tAsA, tCrA_view);
    cute::copy(s2r_tiled_copy_b, tBsB, tCrB_view);

    __syncthreads();

    int num_tile_m = size<1>(tCrC);
    int num_tile_n = size<2>(tCrC);

    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

    for (int m_idx = 0; m_idx < num_tile_m; m_idx++) {
        for (int n_idx = 0; n_idx < num_tile_n; n_idx++) {
            auto cur_src_tensor = tCrC(_, m_idx, n_idx);
            auto cur_dst_tensor = tCgC(_, m_idx, n_idx);

            reduce_add(&cur_dst_tensor(0), cur_src_tensor(0));
            reduce_add(&cur_dst_tensor(1), cur_src_tensor(1));
            reduce_add(&cur_dst_tensor(2), cur_src_tensor(2));
            reduce_add(&cur_dst_tensor(3), cur_src_tensor(3));
        }
    }
}

void invokeFusedQuantizeAndLoraDownSimple(float*       Cptr,
                                          uint32_t*    quantized_input,
                                          float*       quantized_scale,
                                          const void*  Aptr,
                                          const void*  Bptr,
                                          const float* Smoothptr,
                                          int          m,
                                          int          n,
                                          int          k,
                                          int          group_dim,
                                          cudaStream_t stream)
{

    int groups = k / group_dim;
    // warm up
    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;  // bf16 gemm core （写死，因为 flux 暂时只有 bf16）
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kMmaEURepeatM = 2;
    static constexpr int kMmaEURepeatN = 2;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape        = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = 1 * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 4 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT =
        decltype(make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;

    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    constexpr int kTileM = 256;
    constexpr int kTileN = 32;
    constexpr int kTileK = 64;

    assert(kTileN == n);
    assert(kTileK == group_dim);

    // cout << "MMA size: " << size(MMA{}) << endl;
    // cout << " stage2_m: " << stage2_m << " stage2_n: " << stage2_n << " stage2_k: " << stage2_k << endl;
    // cute::print(MMA{});
    // cute::print_latex(MMA{});

    dim3 block(size(MMA{}));
    dim3 grid(k / kTileK, m / kTileM);

    auto SmemLayoutA      = make_layout(make_shape(Int<kTileM>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}));
    auto SmemLayoutB      = make_layout(make_shape(Int<kTileN>{}, Int<kTileK>{}), make_stride(Int<kTileK>{}, Int<1>{}));
    auto SmemLayoutSmooth = make_layout(make_shape(Int<kTileK>{}), make_stride(Int<1>{}));

    int shm_size = (cute::cosize(SmemLayoutA) + cute::cosize(SmemLayoutB)) * sizeof(cute::bfloat16_t)
                   + cute::cosize(SmemLayoutSmooth) * sizeof(float);

    fused_quantized_and_lora_down_simple<kTileM, kTileN, kTileK, MMA>
        <<<grid, block, 0, stream>>>(Cptr, quantized_input, quantized_scale, Aptr, Bptr, Smoothptr, m, n, k);
}

void invokeFusedW4A4GemmAndLoraUp(void*        Dptr,
                                  const void*  Aptr,
                                  const void*  Bptr,
                                  const void*  LoraAptr,
                                  const void*  LoraBptr,
                                  const float* AweightScale,
                                  const float* BweightScale,
                                  int          m,
                                  int          n,
                                  int          k,
                                  int          lora_k,
                                  int          groups,
                                  cudaStream_t stream)
{
    config::GemmConfig<>       gemm_config;
    config::LoraUpGemmConfig<> lora_gemm_config;

    dim3 block(gemm_config.kThreadNum);
    dim3 grid((n + gemm_config.kTileN - 1) / gemm_config.kTileN, (m + gemm_config.kTileM - 1) / gemm_config.kTileM);
    int  shm_size = gemm_config.kShmSize;

    cudaFuncSetAttribute(fused_w4a4_gemm_and_lora_up_multistage_v2<decltype(gemm_config), decltype(lora_gemm_config)>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shm_size);

    fused_w4a4_gemm_and_lora_up_multistage_v2<decltype(gemm_config), decltype(lora_gemm_config)>
        <<<grid, block, shm_size, stream>>>(Dptr,
                                            Aptr,
                                            Bptr,
                                            LoraAptr,
                                            LoraBptr,
                                            AweightScale,
                                            BweightScale,
                                            m,
                                            n,
                                            k,
                                            lora_k,  // lora k
                                            groups);
}

}  // namespace lyradiff