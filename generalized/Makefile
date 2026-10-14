# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
# Makefile for Generalised Recurrence Plot (Marwan 2007b)

CXX      ?= g++
UNAME    := $(shell uname)

# OpenMP flag selection.
#   Linux / Homebrew gcc:  -fopenmp (works for both compile and link)
#   Apple clang on macOS:  needs -Xpreprocessor -fopenmp for the pragma,
#                          plus libomp from Homebrew for the runtime.
IS_APPLE_CLANG := $(shell $(CXX) --version 2>/dev/null | grep -c "Apple clang")

ifeq ($(UNAME),Darwin)
  ifeq ($(IS_APPLE_CLANG),1)
    LIBOMP_PREFIX ?= $(shell brew --prefix libomp 2>/dev/null)
    OPENMP_FLAGS  ?= -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
    OPENMP_LIBS   ?= -L$(LIBOMP_PREFIX)/lib -lomp
  else
    OPENMP_FLAGS  ?= -fopenmp
    OPENMP_LIBS   ?=
  endif
else
  OPENMP_FLAGS ?= -fopenmp
  OPENMP_LIBS  ?=
endif

CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -Wno-unused-parameter $(OPENMP_FLAGS) $(CPPFLAGS)
LDFLAGS  := $(OPENMP_FLAGS) $(OPENMP_LIBS)

SRC := GeneralizedRecurrencePlot.cpp \
       SpatialRQA.cpp \
       SpatialRecurrenceNetwork.cpp \
       JointGeneralizedRecurrencePlot.cpp \
       CrossGeneralizedRecurrencePlot.cpp \
       SpatialSlidingWindow.cpp
OBJ := $(SRC:.cpp=.o)
HDR := GeneralizedRecurrencePlot.h \
       SpatialRQA.h \
       SpatialRecurrenceNetwork.h \
       JointGeneralizedRecurrencePlot.h \
       CrossGeneralizedRecurrencePlot.h \
       SpatialSlidingWindow.h

TEST_TARGET     := test_grp
TEST_SRC        := test_grp.cpp
TEST_OBJ        := $(TEST_SRC:.cpp=.o)

NET_TARGET      := test_network
NET_SRC         := test_network.cpp
NET_OBJ         := $(NET_SRC:.cpp=.o)

JCW_TARGET      := test_joint_cross_window
JCW_SRC         := test_joint_cross_window.cpp
JCW_OBJ         := $(JCW_SRC:.cpp=.o)

all: $(TEST_TARGET) $(NET_TARGET) $(JCW_TARGET)

$(TEST_TARGET): $(OBJ) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(NET_TARGET): $(OBJ) $(NET_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(JCW_TARGET): $(OBJ) $(JCW_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HDR)
	$(CXX) $(CXXFLAGS) -c $<

run: $(TEST_TARGET)
	./$(TEST_TARGET)

run_small: $(TEST_TARGET)
	./$(TEST_TARGET) 32

run_medium: $(TEST_TARGET)
	./$(TEST_TARGET) 64

run_network: $(NET_TARGET)
	./$(NET_TARGET) 32

run_jcw: $(JCW_TARGET)
	./$(JCW_TARGET) 32

clean:
	rm -f $(OBJ) $(TEST_OBJ) $(NET_OBJ) $(JCW_OBJ) \
	      $(TEST_TARGET) $(NET_TARGET) $(JCW_TARGET)

.PHONY: all run run_small run_medium run_network run_jcw clean
