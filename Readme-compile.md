# cmake build
```shell
export ANDROID_NDK_ROOT=/home/lili-5090/Sean/Hexagon_SDK/6.4.0.2/tools/android-ndk-r25c

export OPENCL_SDK_ROOT=/home/lili-5090/Sean/Hexagon_SDK/6.4.0.2/tools/android-ndk-r25c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android
export HEXAGON_SDK_ROOT=/home/lili-5090/Sean/Hexagon_SDK/6.4.0.2/
export HEXAGON_TOOLS_ROOT=/home/lili-5090/Sean/Hexagon_SDK/6.4.0.2/tools/HEXAGON_Tools/19.0.04/

cmake \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DANDROID_PLATFORM=android-31 \
-DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
-DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
-DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
-DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT \
-DPREBUILT_LIB_DIR=android_aarch64 \
-DGGML_OPENMP=OFF \
-DGGML_LLAMAFILE=OFF \
-DGGML_OPENCL=ON \
-DGGML_HEXAGON=ON \
-DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128 \
-DLLAMA_OPENSSL=OFF \
-B build-snapdragon

#-DCMAKE_PREFIX_PATH=$OPENCL_SDK_ROOT \ 

#"ANDROID_ABI":      "arm64-v8a",
#"ANDROID_PLATFORM": "android-31",
#"CMAKE_TOOLCHAIN_FILE": "$env{ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake",
#"CMAKE_C_FLAGS":   "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE",
#"CMAKE_CXX_FLAGS": "-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE",
#"CMAKE_C_FLAGS_RELEASE":          "-O3 -DNDEBUG",
#"CMAKE_CXX_FLAGS_RELEASE":        "-O3 -DNDEBUG",
#"CMAKE_C_FLAGS_RELWITHDEBINFO":   "-O3 -DNDEBUG -g",
#"CMAKE_CXX_FLAGS_RELWITHDEBINFO": "-O3 -DNDEBUG -g",
#"CMAKE_PREFIX_PATH":  "$env{OPENCL_SDK_ROOT}",
#"HEXAGON_SDK_ROOT":   "$env{HEXAGON_SDK_ROOT}",
#"HEXAGON_TOOLS_ROOT": "$env{HEXAGON_TOOLS_ROOT}",
#"PREBUILT_LIB_DIR": "android_aarch64",
#"GGML_OPENMP":      "OFF",
#"GGML_LLAMAFILE":   "OFF",
#"GGML_OPENCL":      "ON",
#"GGML_HEXAGON":     "ON",
#"GGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE": "128",
#"LLAMA_OPENSSL":    "OFF"

cmake --build build-snapdragon --config Release -j 22

```

# Compile error
/home/lili-5090/Sean/llama.cpp/tools/server/server-http.h:72:18: error: no template named 'unordered_map' in namespace 'std' mutable std::unordered_map<std::string, handler_t> handlers;

server-http.h add header file.
++ #include <unordered_map>

```shell
mkdir snapdragon
cmake --install build-snapdragon --prefix hmx-hvx/ --config Release

adb push hmx-hvx/ /data/local/tmp/
```

# How to Run
```shell
adb shell
cd /data/local/tmp/snapdragon/
# M=Llama-3.2-1B-Instruct-Q4_0.gguf D=HTP0 ./scripts/snapdragon/adb/run-cli.sh -no-cnv -p "what is the most popular cookie in the world?"

LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/llama-3.2-3b-instruct.q8_0.gguf \
--poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 \
--ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on \
-ngl 99 --device HTP0 -no-cnv -p "what is the most popular cookie in the world?"

LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/llama-3.2-3b-instruct.q8_0.gguf \
--poll 1000 -t 6 --cpu-mask 0xfc --cpu-strict 1 \
--ctx-size 8192 --batch-size 128 -ctk q8_0 -ctv q8_0 -fa on \
-ngl 99 --device HTP0 -no-cnv -f ../models/fix-token.txt --no-display-prompt

LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/llama-3.2-3b-instruct.q8_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt

# compare with gpu
LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/llama-3.2-3b-instruct.q8_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device GPUOpenCL -f ../models/fix-token.txt --no-display-prompt
```

# How to Run MoE
```shell
LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt

LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/granite-3.0-1b-a400m-instruct-Q8_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt

# compare with gpu
LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device GPUOpenCL -f ../models/fix-token.txt --no-display-prompt

LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/granite-3.0-1b-a400m-instruct-Q8_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on \
-ngl 99 --device GPUOpenCL -f ../models/fix-token.txt --no-display-prompt
```

# NPU Profiling
```shell
cmake \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DANDROID_PLATFORM=android-31 \
-DCMAKE_C_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
-DCMAKE_CXX_FLAGS="-march=armv8.7a+fp16+dotprod+i8mm -fvectorize -ffp-model=fast -fno-finite-math-only -flto -D_GNU_SOURCE" \
-DHEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT \
-DHEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT \
-DPREBUILT_LIB_DIR=android_aarch64 \
-DGGML_OPENMP=OFF \
-DGGML_LLAMAFILE=OFF \
-DGGML_OPENCL=ON \
-DGGML_HEXAGON=ON \
-DGGML_HEXAGON_FP32_QUANTIZE_GROUP_SIZE=128 \
-DGGML_HEXAGON_VERBOSE=1 \
-DLLAMA_OPENSSL=OFF \
-B build-snapdragon


GGML_HEXAGON_VERBOSE=1 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt

GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt

--device GPUOpenCL
--device HTP0,GPUOpenCL
--device GPUOpenCL,HTP0

GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 \
LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 -n 128 --device GPUOpenCL,HTP0 -f ../models/fix-token.txt --no-display-prompt

GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device GPUOpenCL -f ../models/fix-token.txt --no-display-prompt

GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device HTP0,GPUOpenCL -f ../models/fix-token.txt --no-display-prompt

GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-cli --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device GPUOpenCL,HTP0 -f ../models/fix-token.txt --no-display-prompt

LLAMA_LOG_VERBOSITY=4 \
GGML_SCHED_DEBUG=2 \
GGML_HEXAGON_VERBOSE=1 \
GGML_HEXAGON_PROFILE=1 \
LD_LIBRARY_PATH=lib \
ADSP_LIBRARY_PATH=lib \
./bin/llama-completion \
    -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
    --device HTP0 \
    -ngl 99 \
    -c 1024 \
    -b 128 \
    -n 32 \
    -p "Explain sparse matrix multiplication." \
    2>&1 | tee hexagon-placement.log
```


```shell

nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=none \
    -o moe_profile \
    ./build/bin/llama-cli \
    -m /data/deepseek-v2-lite-chat-q4_0.gguf \
    --n-cpu-moe 30 \
    -f ../SmartOrchV2/fix-token.txt \
    -n 32
```
```shell    
nsys stats \
    --report cuda_gpu_mem_size_sum \
    moe_profile.nsys-rep 
    
 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)   Count  Avg (MB)   Med (MB)  Min (MB)   Max (MB)   StdDev (MB)            Operation           
 -----------  -----  ---------  --------  --------  ----------  -----------  ------------------------------
 109,565.712     56  1,956.531     0.000     0.000  27,391.427    7,118.213  [CUDA memset]                 
   8,030.877  1,703      4.716     0.049     0.000     172.032       13.588  [CUDA memcpy Host-to-Device]  
      21.829  1,777      0.012     0.008     0.000       0.410        0.056  [CUDA memcpy Device-to-Host]  
       0.442     54      0.008     0.008     0.008       0.008        0.000  [CUDA memcpy Device-to-Device]

```

```shell  
nsys stats \
    --report cuda_gpu_mem_time_sum \
    --timeunit msec \
    moe_profile.nsys-rep

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ms)  Count  Avg (ms)  Med (ms)  Min (ms)  Max (ms)  StdDev (ms)            Operation           
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     89.6         572.8898  1,703    0.3364    0.0028    0.0003   14.4329       1.0181  [CUDA memcpy Host-to-Device]  
     10.2          65.1686     56    1.1637    0.0003    0.0003   16.3312       4.2322  [CUDA memset]                 
      0.2           1.1987  1,777    0.0007    0.0005    0.0003    0.0184       0.0017  [CUDA memcpy Device-to-Host]  
      0.0           0.0442     54    0.0008    0.0008    0.0008    0.0009       0.0000  [CUDA memcpy Device-to-Device]

```

```shell 
nsys stats \
--report cuda_gpu_trace \
--timeunit msec \
moe_profile.nsys-rep
```


# GGML_OPENCL_PROFILING to profiling kernel launch
```shell 
cmake \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
-DANDROID_ABI=arm64-v8a \
-DANDROID_PLATFORM=android-28 \
-DBUILD_SHARED_LIBS=OFF \
-DLLAMA_CURL=OFF \
-DGGML_OPENCL=ON \
-DGGML_OPENMP=OFF \
-B build-android

cmake --build build-android --config Release -j 22

mkdir bandwidth

cmake --install build-android --prefix multiex/ --config Release

adb push multiex/ /data/local/tmp/

LD_LIBRARY_PATH=lib ./bin/llama-completion -m ../models/deepseek-v2-lite-chat-q4_0.gguf -n 10 -no-cnv -f ../models/fix-token.txt -ngl 30 -v

W=OpenCL#ffn_moe_down-9#0 bytes=6635520 queue=240.000 us submit=27.000 us transfer=10.000 us ocl_total=277.000 us cpu_wall=281.000 us BW=663.552 GB/s
W=OpenCL#ffn_moe_down-9#0 bytes=49152 queue=5352.400 us submit=385.700 us transfer=8.400 us ocl_total=5746.500 us cpu_wall=5825.000 us BW=5.851 GB/s
```

LD_LIBRARY_PATH=lib ./bin/llama-server -m ../models/deepseek-v2-lite-chat-q4_0.gguf

adb forward tcp:8080 tcp:8080
adb forward --remove tcp:8080

## Modify tile size
GGML_OPENCL_MOE_TILE_N
#define TILESIZE_N 32

## tile compute time
GGML_OPENCL_MOE_PROFILE_WARMUP=2 GGML_OPENCL_MOE_PROFILE_MAX_CALLS=120 GGML_OPENCL_MOE_PROFILE_DETAIL=1 LD_LIBRARY_PATH=lib ./bin/llama-completion -m ../models/deepseek-v2-lite-chat-q4_0.gguf -n 10 -no-cnv -f ../models/fix-token.txt -ngl 30 

## tile load time
export GGML_OPENCL_Q4_0_MOE_DP4A=0
export GGML_OPENCL_MOE_WEIGHT_PROFILE_REPEAT=10

GGML_OPENCL_MOE_WEIGHT_PROFILE_WARMUP=0 GGML_OPENCL_MOE_WEIGHT_PROFILE_MAX_CALLS=120 GGML_OPENCL_MOE_WEIGHT_PROFILE=1 LD_LIBRARY_PATH=lib ./bin/llama-completion -m ../models/deepseek-v2-lite-chat-q4_0.gguf -n 10 -no-cnv -f ../models/fix-token.txt -ngl 30

# run SnapdragonProfiler
./run_sdp.sh


# GPU-NPU co-execution
```c++
if ((strstr(src0->name, "as") != NULL) || backend_ctx->toggle_reorder) {
    moe_router_reoerder(backend, src2, ne20);
    backend_ctx->toggle_reorder = false;
}

# Modify 
moe_router_reoerder(backend, src2, ne20);
backend_ctx->toggle_reorder = false;
```
LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib \
./bin/llama-completion \
-m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--device HTP0,GPUOpenCL \
--split-mode layer \
--tensor-split 1,0 \
-ngl 99 \
-ot 'blk\.\d+\.ffn_(up|down|gate_up|gate)_(ch|)exps=OpenCL,blk\.\d+\.ffn_(up|down|gate)_shexp=OpenCL' \
-c 4096 \
-b 128 \
-ub 128 -f ../models/fix-token.txt --no-display-prompt -v




GGML_HEXAGON_VERBOSE=1 GGML_HEXAGON_PROFILE=1 GGML_SCHED_DEBUG=2 LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib ./bin/llama-completion --no-mmap -m ../models/deepseek-v2-lite-chat-q4_0.gguf \
--ctx-size 8192 --batch-size 128 -fa on -v \
-ngl 99 --device HTP0 -f ../models/fix-token.txt --no-display-prompt -n 1 > llama.log 2>&1


# FARF log output 
adb logcat -c
adb logcat -v time -s adsproc


# Perfetto + FrameTimeline
frame_test.pbtxt

adb push frame_test.pbtxt /data/misc/perfetto-configs/

adb shell pm list packages | grep example

adb shell monkey -p com.example.game 1

adb shell perfetto \
--txt \
-c /data/misc/perfetto-configs/frame_test.pbtxt \
-o /data/misc/perfetto-traces/frame_test.perfetto-trace

adb pull /data/misc/perfetto-traces/frame_test.perfetto-trace

https://ui.perfetto.dev/

## SQL
```sql
SELECT * FROM actual_frame_timeline_slice;

######################### JANK RATE #########################
SELECT
COUNT(*) AS total_frames,

    SUM(
      CASE
        WHEN jank_type != 'None'
        THEN 1
        ELSE 0
      END
    ) AS janky_frames,

    100.0 *
    SUM(
      CASE
        WHEN jank_type != 'None'
        THEN 1
        ELSE 0
      END
    ) / COUNT(*) AS jank_percent

FROM actual_frame_timeline_slice;


######################### AVG FPS #########################
WITH frames AS (
    SELECT ts
    FROM actual_frame_timeline_slice
),
     range AS (
         SELECT
             MIN(ts) AS start_ts,
             MAX(ts) AS end_ts,
             COUNT(*) AS frame_count
         FROM frames
     )
SELECT
    frame_count,
    (end_ts - start_ts) / 1e9 AS duration_sec,
    frame_count / ((end_ts - start_ts) / 1e9) AS avg_fps
FROM range;

###################### ONE LOW FPS #########################
WITH frames AS (
    SELECT ts
    FROM actual_frame_timeline_slice
),
     fps_per_second AS (
         SELECT
             CAST(ts / 1000000000 AS INT) AS second,
    COUNT(*) AS fps
FROM frames
GROUP BY second
    ),
    ranked AS (
SELECT
    fps,
    PERCENT_RANK() OVER (ORDER BY fps ASC) AS pct
FROM fps_per_second
    )
SELECT
    AVG(fps) AS one_percent_low_fps
FROM ranked
WHERE pct <= 0.01;
```

# 4x8 Multi expert on one Tile 
GGML_OPENCL_Q4_0_MOE_4X8=1
GGML_OPENCL_Q4_0_MOE_2X16=1
GGML_OPENCL_Q4_0_MOE_4X8=1

