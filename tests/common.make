LIBXC_INSTALL_DIR    = /home/mkurisu/Workspace/libxc/install
H2PERI_INSTALL_DIR   = ..
SIMINT_INSTALL_DIR   = /gpfs/projects/JiaoGroup/hongjigao/simint/build-avx512/install
YATDFT_INSTALL_DIR   = /gpfs/projects/JiaoGroup/hongjigao/YATDFT_forMP2
OPENBLAS_INSTALL_DIR = /gpfs/software/openblas/0.3.21/

DEFS    = 
INCS    = -I$(H2PERI_INSTALL_DIR)/include -I$(YATDFT_INSTALL_DIR)/include -I$(SIMINT_INSTALL_DIR)/include
CFLAGS  = $(INCS) -Wall -g -std=gnu11 -O3 -fPIC $(DEFS)
LDFLAGS = -g -O3 -fopenmp
LIBS    = $(H2PERI_INSTALL_DIR)/lib/libH2PERI.a $(YATDFT_INSTALL_DIR)/lib/libYATDFT.a $(SIMINT_INSTALL_DIR)/lib64/libsimint.a 

ifeq ($(shell $(CC) --version 2>&1 | grep -c "icc"), 1)
CFLAGS  += -fopenmp -xHost
endif

ifeq ($(shell $(CC) --version 2>&1 | grep -c "gcc"), 1)
CFLAGS  += -fopenmp -march=native
LIBS    += -lgfortran -lm
endif

ifeq ($(strip $(USE_MKL)), 1)
DEFS    += -DUSE_MKL
CFLAGS  += -mkl
LDFLAGS += -mkl
endif

ifeq ($(strip $(USE_OPENBLAS)), 1)
DEFS    += -DUSE_OPENBLAS
INCS    += -I$(OPENBLAS_INSTALL_DIR)/include
LDFLAGS += -L$(OPENBLAS_INSTALL_DIR)/lib
LIBS    += -lopenblas
endif

ifeq ($(USE_LIBXC), 1)
CFLAGS  += -DUSE_LIBXC
INCS    += -I$(LIBXC_INSTALL_DIR)/include
LDFLAGS += -L$(LIBXC_INSTALL_DIR)/lib -lxc
endif

C_SRCS 	= $(wildcard *.c)
C_OBJS  = $(C_SRCS:.c=.c.o)
EXES    = $(C_SRCS:.c=.exe)

# Delete the default old-fashion double-suffix rules
.SUFFIXES:

.SECONDARY: $(C_OBJS)

all: $(EXES)

%.c.o: %.c
	$(CC) $(CFLAGS) -c $^ -o $@

%.exe: %.c.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(EXES) $(C_OBJS)
	
	rm -rf ur*
	rm -rf uc*
