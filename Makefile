CC ?= cc
CFLAGS ?= -O2
WARNFLAGS := -std=c11 -Wall -Wextra -Wpedantic -Werror -Iinclude
LDFLAGS ?=
LDLIBS := -lm
CUDA ?= OFF
BUILD := build
ifeq ($(CUDA),ON)
  NVCC ?= nvcc
  CUDA_ARCH ?= 89
  CUDA_OBJ := $(BUILD)/cuda_backend.o
  WARNFLAGS += -DUA_WITH_CUDA=1
  LDLIBS += -lcudart -lstdc++
endif

.PHONY: all clean test sanitize
all: $(BUILD)/uniad

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) $(WARNFLAGS) -c $< -o $@

$(BUILD)/cuda_backend.o: src/cuda_backend.cu | $(BUILD)
	$(NVCC) -O2 -arch=sm_$(CUDA_ARCH) -Iinclude -c $< -o $@

$(BUILD)/uniad: $(BUILD)/uniad.o $(BUILD)/operators.o $(BUILD)/cli.o $(CUDA_OBJ)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

test: all
	$(CC) $(CFLAGS) $(WARNFLAGS) -Isrc tests/test_runtime.c $(BUILD)/uniad.o $(BUILD)/operators.o $(CUDA_OBJ) $(LDLIBS) -o $(BUILD)/test_runtime
	$(BUILD)/test_runtime
	python3 tests/test_cli.py
	python3 tools/oracle.py --compare $(BUILD)/uniad --asset-dir $(BUILD)/oracle-demo
	python3 tools/validate_site.py

sanitize:
	$(MAKE) clean
	$(MAKE) CFLAGS="-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer" LDFLAGS="-fsanitize=address,undefined"
	$(CC) $(CFLAGS) $(WARNFLAGS) -Isrc tests/test_runtime.c $(BUILD)/uniad.o $(BUILD)/operators.o -lm -fsanitize=address,undefined -o $(BUILD)/test_runtime
	ASAN_OPTIONS=detect_leaks=1 $(BUILD)/test_runtime

clean:
	rm -rf $(BUILD)
