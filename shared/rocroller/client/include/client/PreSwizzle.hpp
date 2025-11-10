/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#pragma once

#include <vector>

#include <rocRoller/TensorDescriptor.hpp>

#include <client/GEMMParameters.hpp>

namespace rocRoller::Client
{
    /**
     * @brief Pre-swizzle and optionally pre-tile the input.
     *
     * This assumes that the incoming TensorDescriptor `desc` has been
     * created with `withNormalizedDimensions()`.  That is, the
     * left-most dimension (the 0 dimension) is the fastest dimension
     * (has the smallest stride).
     */
    template <typename T>
    inline std::vector<T> preSwizzle(std::vector<T> const&      input,
                                     TensorDescriptor const&    desc,
                                     std::vector<size_t> const& preSwizzleSize,
                                     std::vector<size_t> const& preTileSize)
    {
        if(not preSwizzleSize.empty())
        {
            AssertFatal(preSwizzleSize.size() == 3,
                        ShowValue(preSwizzleSize.size()),
                        ShowValue(preSwizzleSize));
        }
        AssertFatal(desc.dimensions() == 2,
                    "Batch dimension not yet supported.",
                    ShowValue(desc.dimensions()),
                    ShowValue(desc));
        AssertFatal(desc.totalAllocatedElements() == input.size(),
                    ShowValue(desc),
                    ShowValue(input.size()));

        std::vector<size_t> srcSizes, dimOrder;

        if((not preSwizzleSize.empty()) && (preTileSize.empty()))
        {
            auto tileMN   = preSwizzleSize[0];
            auto tileK    = preSwizzleSize[1];
            auto subTileK = preSwizzleSize[2];

            size_t instPerTileK   = tileK / subTileK;
            size_t instKPerTileMN = tileMN / subTileK;

            srcSizes = {subTileK,
                        instPerTileK,
                        desc.size(0) / (tileK),
                        instKPerTileMN,
                        subTileK,
                        desc.size(1) / (tileMN)};
            dimOrder = {4, 1, 2, 3, 0, 5};
        }
        else if((preSwizzleSize.empty()) && (not preTileSize.empty()))
        {
            srcSizes = {preTileSize[0],
                        desc.size(0) / preTileSize[0], // Number of fast tiles
                        preTileSize[1],
                        desc.size(1) / preTileSize[1]}; // Number of slow tiles

            // Pre-tiling: 1 and 3 are pushed to the back (they become the slowest)
            //
            //   { 0, 2, 1, 3 }
            //

            Log::info("PRE TILE ONLY {}", srcSizes);

            dimOrder = {0, 2, 1, 3};
        }
        else
        {
            Log::warn("Pre-swizzling and pre-tiling are both enabled.  This is not yet tested.");
	    Log::warn("Sizes: preSwizzleSize={}, preTileSize={}", preSwizzleSize, preTileSize);

            AssertFatal(preTileSize[0] >= preSwizzleSize[2],
                        "Pre-tile size must be larger than preSwizzle tile-size",
                        ShowValue(preTileSize[0]),
                        ShowValue(preSwizzleSize[2]));
            AssertFatal(preTileSize[1] >= preSwizzleSize[0],
                        "Pre-tile size must be larger than preSwizzle tile-size",
                        ShowValue(preTileSize[1]),
                        ShowValue(preSwizzleSize[0]));

            AssertFatal(preTileSize[0] % preSwizzleSize[2] == 0,
                        "Pre-tile and pre-swizzle tile-size mismatch.",
                        ShowValue(preTileSize[0]),
                        ShowValue(preSwizzleSize[2]));
            AssertFatal(preTileSize[1] % preSwizzleSize[0] == 0,
                        "Pre-tile and pre-swizzle tile-size mismatch.",
                        ShowValue(preTileSize[1]),
                        ShowValue(preSwizzleSize[0]));

            srcSizes = {preSwizzleSize[2], // Pre-swizzling (swapped)
                        preTileSize[0] / preSwizzleSize[2], // Pre-tiling
                        desc.size(0) / preTileSize[0], // Number of fast tiles
                        preSwizzleSize[0] / preSwizzleSize[2], // Pre-swizzling (but not swapped)
                        preSwizzleSize[2], // Pre-swizzling (swapped)
                        preTileSize[1] / preSwizzleSize[0], // Pre-tiling
                        desc.size(1) / preTileSize[1]}; // Number of slow tiles

            // Pre-tiling: 2 and 6 are pushed to the back (they become the slowest)
            //
            //   { 0, 1, 3, 4, 5, 2, 6 }
            //
            // Pre-swizzling: 4 and 0 are swapped
            //
            //   { 4, 1, 3, 0, 5, 2, 6 }

            dimOrder = {4, 1, 3, 0, 5, 2, 6};
        }

        TensorDescriptor src(desc.dataType(), srcSizes);

        AssertFatal(src.totalAllocatedElements() == desc.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(desc.totalAllocatedElements()),
                    ShowValue(src.totalAllocatedElements() / desc.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(desc));

        auto dst = TensorDescriptor::ShuffledNoPadding(desc.dataType(), srcSizes, dimOrder);

        AssertFatal(src.totalAllocatedElements() == dst.totalAllocatedElements(),
                    ShowValue(src.totalAllocatedElements()),
                    ShowValue(dst.totalAllocatedElements()),
                    ShowValue(src),
                    ShowValue(dst));

        return shuffleDims(input, dst, src);
    }

}
