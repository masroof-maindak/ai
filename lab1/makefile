cc=g++
CFLAGS=-std=c++20
DEBUG=-g -O0
TARGET=bscs22012_src

Default:
	$(cc) $(CFLAGS) $(TARGET).cpp && ./a.out

debug:
	CFLAGS += DEBUG
	$(cc) $(CFLAGS) $(TARGET).cpp
