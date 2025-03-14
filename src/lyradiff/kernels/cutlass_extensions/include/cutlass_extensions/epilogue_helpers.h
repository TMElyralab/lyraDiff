

#pragma once

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

namespace lyradiff {

namespace cutlass_extensions {

struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpDefaultSilu {};

struct EpilogueOpDefaultReLU {};

struct EpilogueOpDefaultFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpDefault {};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {};

constexpr auto BiasScaleMode = cutlass::epilogue::thread::ScaleType::NoBetaScaling;

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                BiasScaleMode>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                BiasScaleMode>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasFtGelu> {
    using Op = cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor,
                                                                   ElementType,
                                                                   ElementsPerVectorAccess,
                                                                   ElementAccumulator,
                                                                   ElementAccumulator,
                                                                   BiasScaleMode,
                                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                                   true>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
    using Op = cutlass::epilogue::thread::
        LinearCombination<ElementType, ElementsPerVectorAccess, ElementAccumulator, ElementAccumulator, BiasScaleMode>;
};

constexpr auto DefaultScaleMode = cutlass::epilogue::thread::ScaleType::Default;

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultSilu> {
    using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                DefaultScaleMode>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultReLU> {
    using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType,
                                                                ElementsPerVectorAccess,
                                                                ElementAccumulator,
                                                                ElementAccumulator,
                                                                DefaultScaleMode>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultFtGelu> {
    using Op = cutlass::epilogue::thread::LinearCombinationGeneric<cutlass::epilogue::thread::GELU_taylor,
                                                                   ElementType,
                                                                   ElementsPerVectorAccess,
                                                                   ElementAccumulator,
                                                                   ElementAccumulator,
                                                                   DefaultScaleMode,
                                                                   cutlass::FloatRoundStyle::round_to_nearest,
                                                                   true>;
};

template<typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefault> {
    using Op = cutlass::epilogue::thread::LinearCombination<ElementType,
                                                            ElementsPerVectorAccess,
                                                            ElementAccumulator,
                                                            ElementAccumulator,
                                                            DefaultScaleMode>;
};

}  // namespace cutlass_extensions
}  // namespace lyradiff
