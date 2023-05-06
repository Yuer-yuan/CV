TARGET_EXEC ?= ratio_test

BUILD_DIR := ./build
SRC_DIR := ./src
INC_DIR := $(shell find $(SRC_DIR) -type d)

INC_FLAGS := $(addprefix -I,$(INC_DIR))
SRCS := $(shell find $(SRC_DIR) -name *.cpp -type f)
OBJS := $(SRCS:%.cpp=$(BUILD_DIR)/%.o)

# CXX := g++
CXXFLAGS := -std=c++17 $(INC_FLAGS)
CXXFLAGS := -std=c++17
MODE ?= release

# OPENCV_CFLAGS := `pkg-config --cflags opencv`
OPENCV_CFLAGS := -I/usr/local/include/opencv4
# OPENCV_LIBS := `pkg-config --libs opencv`
OPENCV_LIBS := -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_highgui -lopencv_imgcodecs -lopencv_xfeatures2d -lopencv_features2d -lopencv_calib3d -lopencv_flann
CXXFLAGS += $(OPENCV_CFLAGS)
LDFLAGS := $(OPENCV_LIBS)

# defines
INTERACTIVE := 0
DEFINES :=
ifeq ($(INTERACTIVE), 1)
	DEFINES := -DINTERACTIVE
endif
CXXFLAGS += $(DEFINES)

# main args
IMG_1 := ./assets/test1.jpg
IMG_2 := ./assets/test2.jpg
RATIO := 6
PRECISION := 10
SAVE_DIR := $(BUILD_DIR)/save
MAIN_ARGS := --img_1 $(IMG_1) --img_2 $(IMG_2) -r $(RATIO) -s $(SAVE_DIR) -p $(PRECISION)

.PHONY: all
all: $(BUILD_DIR)/$(TARGET_EXEC)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	@$(CXX) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@mkdir -p $(SAVE_DIR)
ifeq ($(MODE), debug)
	@$(CXX) $(CXXFLAGS) -g -O0 -Wall -c $< -o $@ 
else
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

.PHONY: help
help: $(BUILD_DIR)/$(TARGET_EXEC)
	@$(BUILD_DIR)/$(TARGET_EXEC) -H