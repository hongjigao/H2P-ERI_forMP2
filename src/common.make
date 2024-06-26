LIB_A   = libH2PERI.a
LIB_SO  = libH2PERI.so

C_SRCS  = $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)

SIMINT_INSTALL_DIR   = /gpfs/projects/JiaoGroup/hongjigao/gccmp2test/simint/build-avx512/install
OPENBLAS_INSTALL_DIR = /gpfs/software/openblas/0.3.21/
YATDFT_INSTALL_DIR   = /gpfs/projects/JiaoGroup/hongjigao/YATDFT_forMP2

DEFS    = 
INCS    = -I$(SIMINT_INSTALL_DIR)/include -I$(YATDFT_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O2 -fPIC $(DEFS)

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
AR      = xiar rcs
CFLAGS += -qopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
AR      = ar rcs
CFLAGS += -fopenmp -march=native
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS   += -DUSE_MKL
CFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS   += -DUSE_OPENBLAS
INCS   += -I$(OPENBLAS_INSTALL_DIR)/include
endif

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: install

install: $(LIB_A) $(LIB_SO)
	mkdir -p ../lib
	mkdir -p ../include
	cp -u $(LIB_A)  ../lib/$(LIB_A)
	cp -u $(LIB_SO) ../lib/$(LIB_SO)
	cp -u *.h ../include/

$(LIB_A): $(C_OBJS) 
	$(AR) $@ $^

$(LIB_SO): $(C_OBJS) 
	$(CC) -shared -o $@ $^

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

clean:
	rm -f $(C_OBJS) $(LIB_A) $(LIB_SO)
