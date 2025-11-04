MPICC ?= mpicc
CFLAGS ?= -std=c11 -Wall -Wextra
LDFLAGS ?=

SRC_DIR := src
BUILD_DIR := build

SOURCES := \
	$(SRC_DIR)/monte_carlo_pi.c \
	$(SRC_DIR)/matvec_rows.c \
	$(SRC_DIR)/matvec_cols.c \
	$(SRC_DIR)/matvec_blocks.c \
	$(SRC_DIR)/matmul_cannon.c

PROGRAMS := $(foreach src,$(SOURCES),$(BUILD_DIR)/$(basename $(notdir $(src))))

.PHONY: all clean lint

all: $(BUILD_DIR) $(PROGRAMS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

define build_rule
$(BUILD_DIR)/$(basename $(notdir $(1))): $(1) | $(BUILD_DIR)
	$$(MPICC) $$(CFLAGS) $$< -o $$@ $$(LDFLAGS)
endef

$(foreach src,$(SOURCES),$(eval $(call build_rule,$(src))))

clean:
	rm -rf $(BUILD_DIR)

lint:
	clang-format --dry-run --Werror $(SOURCES)
