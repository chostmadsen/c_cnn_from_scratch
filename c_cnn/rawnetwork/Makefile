# compiler and flags
CC = clang
CFLAGS = -Iinclude -Wall -Wextra -std=c17 -O2

# dirs
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

# src files and corresponding obj files
SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC))

# out binary
TARGET = main

# default rule
all: $(BUILD_DIR) $(TARGET)

# link the final executable
$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $@

# .c to .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# build dir
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# rebuild
rebuild: clean all

.PHONY: all clean rebuild
