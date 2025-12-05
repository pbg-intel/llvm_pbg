#define HEAP_SIZE                          (0x1400000)
#define NUM_OF_HEAPS                       (1)
#define RANDOM_WALK_LENGTH                 (50)
#define NUM_OF_SUPERBLOCKS_PER_HEAP        (2048)
#define NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK  (32)
#define INCLUDE_UNWINDING                  (1)


// super block structure
union superblk {
  unsigned long SuperblkMetadata;
  struct {
    unsigned char block_enable_partition_info : 8;
    unsigned char partition_block_occupancy_tracker : 8; // bits used for tracking blks
    unsigned short alloc_length_tracker_bits : 16; // bits used for tracking blks
    unsigned int alloc_size : 32; //size of allocation
  } fields;
};

struct random_walk_params_t {
  int num_walks;
  int walk_length;
  int index_range_start;
  int index_range_end;
  int step_size;
  int initial_pos;

  unsigned long long base_seed;
  unsigned long long base_subseed;
};

struct device_heap_t {
  unsigned long base;
  unsigned long size;
  unsigned int blocksize;
  unsigned int max_num_blocks;

  unsigned int max_num_heap_partitions;
  unsigned long *ptr_superblk;
  struct random_walk_params_t *prwalkparams;
};

struct allocator_context_t
{
	struct device_heap_t* deviceheap[NUM_OF_HEAPS];
};