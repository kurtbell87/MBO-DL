cmake_minimum_required(VERSION 3.20)
project(mbo-dl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------- GoogleTest via FetchContent ----------
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ---------- Tests ----------
enable_testing()
include(GoogleTest)

set(TEST_TARGETS
    book_builder_test
    feature_encoder_test
    oracle_labeler_test
    trajectory_builder_test
    gbt_features_test
    bar_builder_test
    day_event_buffer_test
    warmup_test
    execution_costs_test
    oracle_replay_test
    triple_barrier_test
    multi_day_backtest_test
    regime_stratification_test
    oracle_comparison_test
    backtest_criteria_test
    bar_features_test
    raw_representations_test
    feature_export_test
    feature_warmup_test
    feature_mi_test
    conditional_returns_test
    bar_comparison_test
    oracle_expectancy_test
)

foreach(target ${TEST_TARGETS})
    add_executable(${target} tests/${target}.cpp)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${target} PRIVATE GTest::gtest_main)
    gtest_discover_tests(${target})
endforeach()

# ---------- libtorch ----------
find_package(Torch REQUIRED)

# ---------- ONNX Runtime (for serialization header included via training_loop) ----------
find_package(onnxruntime REQUIRED)

# ---------- Tests requiring libtorch ----------
set(TORCH_TEST_TARGETS
    mlp_model_test
    cnn_model_test
)

foreach(target ${TORCH_TEST_TARGETS})
    add_executable(${target} tests/${target}.cpp)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${target} PRIVATE GTest::gtest_main "${TORCH_LIBRARIES}")
    target_compile_options(${target} PRIVATE ${TORCH_CXX_FLAGS})
    gtest_discover_tests(${target})
endforeach()

# ---------- databento-cpp via FetchContent ----------
FetchContent_Declare(
    databento
    GIT_REPOSITORY https://github.com/databento/databento-cpp.git
    GIT_TAG        HEAD
)
FetchContent_MakeAvailable(databento)

# ---------- XGBoost via FetchContent ----------
FetchContent_Declare(
    xgboost
    GIT_REPOSITORY https://github.com/dmlc/xgboost.git
    GIT_TAG        v2.1.3
)
set(BUILD_STATIC_LIB ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(USE_OPENMP OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(xgboost)

# ---------- Tests requiring XGBoost ----------
set(XGBOOST_TEST_TARGETS
    gbt_model_test
    gbt_importance_test
)

foreach(target ${XGBOOST_TEST_TARGETS})
    add_executable(${target} tests/${target}.cpp)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${target} PRIVATE GTest::gtest_main xgboost)
    gtest_discover_tests(${target})
endforeach()

# ---------- Serialization test (requires libtorch + xgboost + onnxruntime) ----------
add_executable(serialization_test tests/serialization_test.cpp)
target_include_directories(serialization_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(serialization_test PRIVATE
    GTest::gtest_main
    "${TORCH_LIBRARIES}"
    xgboost
    onnxruntime::onnxruntime
)
target_compile_options(serialization_test PRIVATE ${TORCH_CXX_FLAGS})
gtest_discover_tests(serialization_test)

# ---------- Integration test (requires libtorch + xgboost + databento) ----------
add_executable(integration_overfit_test tests/integration_overfit_test.cpp)
target_include_directories(integration_overfit_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(integration_overfit_test PRIVATE
    GTest::gtest_main
    "${TORCH_LIBRARIES}"
    xgboost
    databento::databento
)
target_compile_options(integration_overfit_test PRIVATE ${TORCH_CXX_FLAGS})
gtest_discover_tests(integration_overfit_test PROPERTIES LABELS "integration")

# ---------- N=128 overfit test (requires libtorch + xgboost + databento) ----------
add_executable(n128_overfit_test tests/n128_overfit_test.cpp)
target_include_directories(n128_overfit_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(n128_overfit_test PRIVATE
    GTest::gtest_main
    "${TORCH_LIBRARIES}"
    xgboost
    databento::databento
)
target_compile_options(n128_overfit_test PRIVATE ${TORCH_CXX_FLAGS})
gtest_discover_tests(n128_overfit_test PROPERTIES LABELS "integration")

# ---------- Subordination Hypothesis Test (R1 experiment) ----------
add_executable(subordination_test tools/subordination_test.cpp)
target_include_directories(subordination_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(subordination_test PRIVATE databento::databento)

# ---------- Information Decomposition Export (R2 experiment) ----------
add_executable(info_decomposition_export tools/info_decomposition_export.cpp)
target_include_directories(info_decomposition_export PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(info_decomposition_export PRIVATE databento::databento)
