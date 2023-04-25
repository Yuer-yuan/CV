TARGET_EXEC ?= canny

BUILD_DIR := ./build
SRC_DIR := ./src
INC_DIR := $(shell find $(SRC_DIR) -type d)

INC_FLAGS := $(addprefix -I,$(INC_DIR))
SRCS := $(shell find $(SRC_DIR) -name *.cpp -type f)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

CXX := g++
CXXFLAGS := -std=c++17 $(INC_FLAGS)
MODE ?= debug

OPENCV_CFLAGS := -I/usr/local/include/opencv4
OPENCV_LIBS := -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CXXFLAGS += $(OPENCV_CFLAGS)
LDFLAGS := $(OPENCV_LIBS)

# main args
ASSEST_DIR := ./assets
IMG_PATH := $(ASSEST_DIR)/Lenna.png
SAVE_DIR := $(BUILD_DIR)/save
LOW := 15
HIGH := 20
LINEAR := 0	# linear interpolation
INTER := 0	# interactive
LINK := 0 	# predict edge point

MAIN_ARGS = -i $(IMG_PATH) -s $(SAVE_DIR) -l $(LOW) -h $(HIGH)
ifeq ($(LINEAR), 1)
	MAIN_ARGS += -n
endif
ifeq ($(INTER), 1)
	MAIN_ARGS += -a
endif
ifeq ($(LINK), 1)
	MAIN_ARGS += -e
endif

.PHONY: all
all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@mkdir -p $(SAVE_DIR)
ifeq ($(MODE), debug)	# debug mode
	@$(CXX) $(CXXFLAGS) -g -O0 -Wall -c $< -o $@
else					# release mode
	@$(CXX) $(CXXFLAGS) -O2 -c $< -o $@
endif

.PHONY: run
run: $(BUILD_DIR)/$(TARGET_EXEC)
	@$(BUILD_DIR)/$(TARGET_EXEC) $(MAIN_ARGS)

.PHONY: gdb
gdb: $(BUILD_DIR)/$(TARGET_EXEC)
	@gdb --args $(BUILD_DIR)/$(TARGET_EXEC) $(MAIN_ARGS)

.PHONY: clean
clean:
	@rm -r $(BUILD_DIR)

.PHONY: clean_data
clean_data:
	@rm -r $(SAVE_DIR)

.PHONY: clean_all
clean_all:
	@rm -r $(BUILD_DIR)
	@rm -r $(SAVE_DIR)

.PHONY: help
help: $(BUILD_DIR)/$(TARGET_EXEC)
	@$(BUILD_DIR)/$(TARGET_EXEC) -H
