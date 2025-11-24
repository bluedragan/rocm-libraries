/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef BATCHNORM_FUNCTIONS_HPP
#define BATCHNORM_FUNCTIONS_HPP

#include "configuration.hpp"

namespace miopen {
namespace batchnorm {

template <typename OutType, typename InType>
__forceinline__ __device__ __host__ OutType cast(InType in)
{
    if constexpr(std::is_same<OutType, InType>::value)
    {
        return in;
    }
    else if constexpr(std::is_same<OutType, ushort>::value && std::is_same<InType, float>::value)
    {
        return float_to_bfloat16(in);
    }
    else if constexpr(std::is_same<InType, ushort>::value && std::is_same<OutType, float>::value)
    {
        return bfloat16_to_float(in);
    }
    else
    {
        return static_cast<OutType>(in);
    }
}

template <typename FpType, typename FpPrecType>
__forceinline__ __device__ __host__ auto
fpprec4_to_fp4(typename mapped_vector_type<FpPrecType, 4>::type const& val)
{
    return typename mapped_vector_type<FpType, 4>::type(
        cast<FpType>(val.x), cast<FpType>(val.y), cast<FpType>(val.z), cast<FpType>(val.w));
}

template <typename FpPrecType, typename FpType>
__forceinline__ __device__ __host__ auto
fp4_to_fpprec4(typename mapped_vector_type<FpType, 4>::type const& val)
{
    return typename mapped_vector_type<FpPrecType, 4>::type(cast<FpPrecType>(val.x),
                                                            cast<FpPrecType>(val.y),
                                                            cast<FpPrecType>(val.z),
                                                            cast<FpPrecType>(val.w));
}

template <typename FpPrecType, typename FpType, size_t VecSize>
__forceinline__ __device__ __host__ auto
fp_to_fpprec_vec(typename mapped_vector_type<FpType, VecSize>::type const& val)
{
    if constexpr(miopen::batchnorm::config::vectorize)
    {
        return fp4_to_fpprec4<FpPrecType, FpType>(val);
    }
    else
    {
        return cast<FpPrecType, FpType>(val);
    }
}

template <typename FpType, typename FpPrecType, size_t VecSize>
__forceinline__ __device__ __host__ auto
fpprec_to_fp_vec(typename mapped_vector_type<FpPrecType, VecSize>::type const& val)
{
    if constexpr(miopen::batchnorm::config::vectorize)
    {
        return fpprec4_to_fp4<FpType, FpPrecType>(val);
    }
    else
    {
        return cast<FpType, FpPrecType>(val);
    }
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate1(T1& a, T2 const& b)
{
    a += cast<T1>(b);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad1(T& a, T const& b, T const& c, T const& d)
{
    a = fma(b, c, d);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate4(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
    a += cast<T1>(b.z);
    a += cast<T1>(b.w);
}

template <typename T1, typename T2, typename T3, typename T4>
__forceinline__ __device__ __host__ void
_accumulate_mad4(T1& a, T2 const& b, T3 const& c, T4 const& d)
{
    a = fma(cast<T1>(b.x), cast<T1>(c.x), cast<T1>(d));
    a = fma(cast<T1>(b.y), cast<T1>(c.y), cast<T1>(d));
    a = fma(cast<T1>(b.z), cast<T1>(c.z), cast<T1>(d));
    a = fma(cast<T1>(b.w), cast<T1>(c.w), cast<T1>(d));
}

template <typename T>
__forceinline__ __device__ __host__ void
_accumulate_mad(T& a,
                typename mapped_vector_type<T, 4>::type const& b,
                typename mapped_vector_type<T, 4>::type const& c,
                T const& d)
{
    static_assert(miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad4(a, b, c, d);
}

template <typename T>
__forceinline__ __device__ __host__ void _accumulate_mad(T& a, T const& b, T const& c, T const& d)
{
    static_assert(!miopen::batchnorm::config::vectorize && (!miopen::config::layout_nhwc),
                  "_accumulate_mad for this particular arg list is disabled.");
    _accumulate_mad1(a, b, c, d);
}

template <typename FpAccumType>
struct StashUpdater
{
    FpAccumType const mean;
    FpAccumType const variance;
    FpAccumType const expAvgFactor;

    __device__ StashUpdater(FpAccumType m, FpAccumType v, FpAccumType e)
        : mean(m), variance(v), expAvgFactor(e)
    {
    }

    __forceinline__ __device__ void operator()(FpAccumType& runningMean,
                                               FpAccumType& runningVariance) const
    {
        const FpAccumType newRunningMean = fma(-expAvgFactor, runningMean, runningMean);
        runningMean                      = fma(mean, expAvgFactor, newRunningMean);

        const FpAccumType adjust = static_cast<FpAccumType>(
            (miopen::batchnorm::config::nhw == 1)
                ? variance
                : variance * (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) /
                              (static_cast<FpAccumType>(miopen::batchnorm::config::nhw) -
                               FpAccumType{1.0})));

        runningVariance =
            fma(FpAccumType(1.0) - expAvgFactor, runningVariance, expAvgFactor * adjust);
    }
};

template <typename FpAccumType>
struct StashUpdaterPA
{
    FpAccumType const mean;
    FpAccumType const variance;
    FpAccumType const expAvgFactor;

    __device__ StashUpdaterPA(FpAccumType m, FpAccumType v, FpAccumType e)
        : mean(m), variance(v), expAvgFactor(e)
    {
    }

    __forceinline__ __device__ void operator()(FpAccumType& runningMean,
                                               FpAccumType& runningVariance) const
    {
        const FpAccumType newRunningMean = fma(-expAvgFactor, runningMean, runningMean);
        runningMean                      = fma(mean, expAvgFactor, newRunningMean);

        const FpAccumType adjust = static_cast<FpAccumType>(
            (miopen::batchnorm::config::n == 1)
                ? variance
                : variance * (static_cast<FpAccumType>(miopen::batchnorm::config::n) /
                              (static_cast<FpAccumType>(miopen::batchnorm::config::n) -
                               FpAccumType{1.0})));

        runningVariance =
            fma(FpAccumType{1.0} - expAvgFactor, runningVariance, expAvgFactor * adjust);
    }
};

template <typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void saved_stash(FpPrecType_C* __restrict resultSaveMean,
                                            FpPrecType_C* __restrict resultSaveInvVariance,
                                            FpAccumType_C mean,
                                            FpAccumType_C invVariance,
                                            unsigned int channel)
{
    resultSaveMean[channel]        = static_cast<FpPrecType_C>(mean);
    resultSaveInvVariance[channel] = static_cast<FpPrecType_C>(invVariance);
}

template <typename FpAccumType_C, typename FpPrecType_C, typename Updater>
__forceinline__ __device__ void running_stash(FpPrecType_C* __restrict resultRunningMean,
                                              FpPrecType_C* __restrict resultRunningVariance,
                                              Updater const& update,
                                              uint channel)
{
    // Variant 4 is not used any more. There used to be a special updater for that case deleted when
    // porting kernels to HIP.
    static_assert(miopen::batchnorm::config::variant != 4,
                  "running_stash is only compiled when MIO_BN_VARIANT != 4.");

    auto pvt_runMean     = static_cast<FpAccumType_C>(resultRunningMean[channel]);
    auto pvt_runVariance = static_cast<FpAccumType_C>(resultRunningVariance[channel]);

    update(pvt_runMean, pvt_runVariance);

    saved_stash(resultRunningMean, resultRunningVariance, pvt_runMean, pvt_runVariance, channel);
}

} // namespace batchnorm
} // namespace miopen

#endif // BATCHNORM_FUNCTIONS_HPP
