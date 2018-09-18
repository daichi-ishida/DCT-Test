CXX := g++
BUILD_TYPE := Debug
OUTPUT_DIR := output

CXX_DEBUG_FLAGS := -g -O0 -Wall
CXX_RELEASE_FLAGS := -s -O2 

CXXFLAGS := -MMD -MP -std=gnu++14 -fopenmp 
BINDIR := bin
SRCDIR := src
OBJDIR := obj
INCLUDE := -I./include

DEBUG := gdb

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(addprefix $(OBJDIR)/, $(notdir $(SRCS:.cpp=.o)))
DEPS := $(OBJS:.o=.d)

INCLUDE	+= $(shell pkg-config --cflags opencv fftw3)
LIBS := $(shell pkg-config --libs opencv fftw3) -lm
EXECUTABLE	:= main
RM := rm -f

ifeq ($(BUILD_TYPE),Release)
	CXXFLAGS += $(CXX_RELEASE_FLAGS)
else ifeq ($(BUILD_TYPE),Debug)
	CXXFLAGS += $(CXX_DEBUG_FLAGS)
else
	$(error buildtype must be release, debug, profile or coverage)
endif

.PHONY : all
all: $(BINDIR)/$(EXECUTABLE)

$(BINDIR)/$(EXECUTABLE): $(OBJS) 
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	@if [ ! -e $(OUTPUT_DIR) ]; then mkdir -p $(OUTPUT_DIR); fi	
	$(CXX) $(CXXFLAGS) $(INCLUDE) $^ -o $@ $(LIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@if [ ! -e `dirname $@` ]; then mkdir -p `dirname $@`; fi
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

.PHONY : debug
debug :
	@if [ ! -e $(OUTPUT_DIR) ]; then mkdir -p $(OUTPUT_DIR); fi
	$(DEBUG) $(BINDIR)/$(EXECUTABLE)

.PHONY : run
run:
	@if [ ! -e $(OUTPUT_DIR) ]; then mkdir -p $(OUTPUT_DIR); fi
	$(BINDIR)/$(EXECUTABLE)

.PHONY : clean
clean:
	$(RM) $(BINDIR)/$(EXECUTABLE) $(OBJS) $(DEPS)

-include $(DEPS)