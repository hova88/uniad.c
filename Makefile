CC ?= cc
CFLAGS ?= -O2
WARNFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -Iinclude
LDFLAGS ?=
LDLIBS := -lm
CUDA ?= OFF
BUILD_ROOT ?= build
BUILD := $(BUILD_ROOT)/cpu
ifeq ($(CUDA),ON)
  NVCC ?= nvcc
  CUDA_ARCH ?= 89
  WARNFLAGS += -DUA_WITH_CUDA=1
  LDLIBS += -lcudart -lstdc++
  BUILD := $(BUILD_ROOT)/cuda
  CUDA_OBJ := $(BUILD)/cuda_backend.o
  TEST_CUDA_ENV := UA_TEST_CUDA=1
endif

.PHONY: all clean test test-core sanitize
all: $(BUILD)/uniad

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(WARNFLAGS) -c $< -o $@

$(BUILD)/cuda_backend.o: src/cuda_backend.cu | $(BUILD)
	$(NVCC) -O2 -arch=sm_$(CUDA_ARCH) -Iinclude -c $< -o $@

$(BUILD)/uniad: $(BUILD)/uniad.o $(BUILD)/operators.o $(BUILD)/sha256.o $(BUILD)/cli.o $(CUDA_OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

test-core: all
	$(CC) $(CFLAGS) $(WARNFLAGS) -Isrc tests/test_runtime.c $(BUILD)/uniad.o $(BUILD)/operators.o $(BUILD)/sha256.o $(CUDA_OBJ) $(LDLIBS) -o $(BUILD)/test_runtime
	$(BUILD)/test_runtime
	UA_TEST_CLI=$(BUILD)/uniad $(TEST_CUDA_ENV) python3 tests/test_cli.py
	python3 tools/validate_site.py

test: test-core
	UA_TEST_CLI=$(BUILD)/uniad python3 tests/test_uaw2.py
	python3 tools/oracle.py --compare $(BUILD)/uniad --asset-dir $(BUILD)/oracle-demo

sanitize:
	$(MAKE) BUILD=build/sanitize CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined"
	$(CC) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(WARNFLAGS) -Isrc tests/test_runtime.c build/sanitize/uniad.o build/sanitize/operators.o build/sanitize/sha256.o -lm -fsanitize=address,undefined -o build/sanitize/test_runtime
	ASAN_OPTIONS=detect_leaks=1 build/sanitize/test_runtime

clean:
	rm -rf $(BUILD)
