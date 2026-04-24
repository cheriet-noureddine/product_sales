CXX = g++
CXXFLAGS = -O2 -std=c++11

SERIAL_DIR = serial
PARALLEL_DIR = parallel

SERIAL_TARGET = svm_serial
PARALLEL_TARGET = svm_parallel

all: $(SERIAL_TARGET) $(PARALLEL_TARGET)

$(SERIAL_TARGET): $(SERIAL_DIR)/main.cpp $(SERIAL_DIR)/svm.cpp $(SERIAL_DIR)/grid_search.cpp
    $(CXX) $(CXXFLAGS) -o $@ $^

$(PARALLEL_TARGET): $(PARALLEL_DIR)/main.cpp $(PARALLEL_DIR)/svm.cpp $(PARALLEL_DIR)/grid_search.cpp
    $(CXX) $(CXXFLAGS) -fopenmp -o $@ $^

clean:
    rm -f $(SERIAL_TARGET) $(PARALLEL_TARGET)
