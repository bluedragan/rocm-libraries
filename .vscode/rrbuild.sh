#!/bin/bash

buildvar="${1:-build}"
rm -rf "$buildvar"
mkdir "$buildvar"
cd "$buildvar"
cmake --preset default:release -B . -S .. -GNinja -DCMAKE_BUILD_TYPE=Debug -DROCROLLER_ENABLE_TIMERS=ON
ninja
