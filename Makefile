CXX       = g++
CXXFLAGS  = -std=c++17 -fopenmp -Ofast -Wno-narrowing -O2
BIN       = ./bin
SRCS      := $(wildcard ./src/*.cpp)
DIR       := $(notdir $(SRCS))
OBJ       := $(patsubst %cpp,./bin/%o,$(DIR))
TARGET    := $(patsubst %cpp,./bin/%a,$(DIR))
RM        = -rm -f
MKL       = -I$(MKLROOT)/include -L$(MKLROOT)/lib -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -liomp5 -lm -ldl

.PHONY: all all-before all-after clean clean-custom

all: all-before $(TARGET) all-after

clean: clean-custom
	$(RM) $(OBJ) $(TARGET)

${BIN}/%.o: ./src/%.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(MKL)

${BIN}/%.a: ${BIN}/%.o
	chmod +x ./bin/nls.py
	chmod +x ./bin/hm.py
	$(CXX) $< -o $@ $(CXXFLAGS) $(MKL)
