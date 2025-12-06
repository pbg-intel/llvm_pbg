#pragma once
#include "spirv_vars.h"
#include "atomic.hpp"
#include "onemkl_philox4x32x10_uniform_bernoulli.hpp"

#define HEAP_SIZE                                  (0x1400000)
#define NUM_OF_HEAPS                               (1)
#define RANDOM_WALK_LENGTH                         (50)
#define NUM_OF_SUPERBLOCKS_PER_HEAP                (2048)
#define NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK          (32)
#define INCLUDE_UNWINDING                          (1)
#define HEAP_10_PERCENT_PARTITION                  (NUM_OF_SUPERBLOCKS_PER_HEAP/10)  // upto 10%
#define HEAP_20_PERCENT_PARTITION                  (HEAP_10_PERCENT_PARTITION * 2)  // upto 20%
#define NUM_PARTITIONS_IN_BLOCK                    (8)
#define FRAGMENTATION_MARGIN_IN_PERCENTAGE         (10) 
// super block structure
union superblk {
  unsigned long SuperblkMetadata;
  struct {
    unsigned char block_enable_partition_info : 8;
    unsigned char partition_block_occupancy_tracker
        : 8; // bits used for tracking blks
    unsigned short alloc_length_tracker_bits : 16; // bits used for tracking
                                                   // blks
    unsigned int alloc_size : 32;

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
  int *prwalk_path;
};

struct allocator_context_t
{
	struct device_heap_t* deviceheap[NUM_OF_HEAPS];
};


int dev_malloc(unsigned long* device_superblk, unsigned int start_blk, unsigned int size, unsigned int end_blk, unsigned int base_blk_size);

DeviceGlobal<void *> __DeviceAllocCtxPtr;

#if defined(__SPIR__) || defined(__SPIRV__)

#define __SYCL_CONSTANT__ __attribute__((opencl_constant))

static const __SYCL_CONSTANT__ char __malloc_prwalk_debug[] =
    "[kernel] random walk params fileds: num_walks=%d, walk_length=%d, "
    "index_range_start=%d, index_range_end=%d, step_size=%d\n";

extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);
extern DEVICE_EXTERNAL int __spirv_ocl_ctz(int) noexcept ;
//extern DEVICE_EXTERNAL int __spirv_ocl_clz(unsigned char) noexcept;
extern DEVICE_EXTERNAL int __spirv_ocl_ctz(unsigned char) noexcept;

//extern SYCL_EXTERNAL int __spirv_AtomicCompareExchange(SYCL_GLOBAL  long long*, int,
 //                                                        int, int, long long,
 //                                                        long long) noexcept;

 //extern SYCL_EXTERNAL int __spirv_AtomicCompareExchange(unsigned long*, int,
//int, int, unsigned long,
//unsigned long) noexcept;

//extern SYCL_EXTERNAL int __spirv_AtomicLoad(unsigned long*, int,
//int) noexcept;

//extern DEVICE_EXTERNAL unsigned long __spirv_AtomicLoad(SPIR_GLOBAL const unsigned long*, int,
//                                              int) noexcept;
DEVICE_EXTERN_C

void *malloc(size_t size) {
  struct allocator_context_t *temp = reinterpret_cast<struct allocator_context_t *> (__DeviceAllocCtxPtr.get());
  device_heap_t *device_heap_ptr = reinterpret_cast<device_heap_t *>(temp->deviceheap);
  random_walk_params_t *prwalk = device_heap_ptr->prwalkparams;

  unsigned long* ptr_superblk = device_heap_ptr->ptr_superblk;

  //id<1> global_id =item.get_global_id(); //need equivalent of this in the compiler..
  int walklength = prwalk->walk_length;
  int walk_id = 1 ; // NHOW TO GET GOLBAL ID FOR WORKITEM ..?global_id; 
  float positive_bias = 0.5f;
  unsigned long alloc_ptr = 0;
  unsigned int num_heap_blks=0;
  int ret_val = 0;
  // Access data directly via USM pointers
  int current_index = prwalk->initial_pos + (NUM_OF_SUPERBLOCKS_PER_HEAP/2) + (walk_id*NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK);

  int index_range_start = prwalk->index_range_start;
  int index_range_end = prwalk->index_range_end;

  int base_seed = prwalk->base_seed+walk_id;
  int base_subseed = prwalk->base_subseed+walk_id;
  int step_size = 1;

  {
    int k = 1;
    while(size > (k * (unsigned int)((HEAP_10_PERCENT_PARTITION)* NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK * device_heap_ptr->blocksize)))
    {
      k++;   
      positive_bias = positive_bias * positive_bias;                     
    }
    int modulo_size = (k > 4) ? 1 : (10 / k) ; // value of 10 indicates number of 10percent segments within the whole heap
    int partition_range  =  NUM_OF_SUPERBLOCKS_PER_HEAP / modulo_size ;

    unsigned int local_id =  (__spirv_BuiltInWorkgroupSize(1) * __spirv_BuiltInWorkgroupSize(0) *    \
                            __spirv_BuiltInLocalInvocationId(2)) +                                 \
                            (__spirv_BuiltInWorkgroupSize(0) *                                      \
                            __spirv_BuiltInLocalInvocationId(1)) +                                 \
                            __spirv_BuiltInLocalInvocationId(0);
                                                        
    int block_range = (modulo_size == 1)? device_heap_ptr->max_num_blocks : (partition_range * NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK);
    index_range_start = (modulo_size == 1) ? 0 : (local_id % modulo_size)*(partition_range * NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK);
    index_range_end = index_range_start + (block_range -1);

    num_heap_blks = (size / device_heap_ptr->blocksize) + 1; // this is the minimum numer of bigger sized blocks required to perform subheap allocation as the sub heap allocation may or may not start on a main block boundary

  }

  oneapi::mkl::rng::device::uniform<int> random_pos(index_range_start, index_range_end);

  oneapi::mkl::rng::device::philox4x32x10 local_engine( base_seed, base_subseed);

  current_index =  oneapi::mkl::rng::device::generate(random_pos,local_engine);   

  int range = ((size/device_heap_ptr->blocksize)) ?  (step_size*0.1*(size/device_heap_ptr->blocksize)) + (size/device_heap_ptr->blocksize) : step_size * 1   ;

for (int i = 0; i < walklength; ++i) 
{

  oneapi::mkl::rng::device::bernoulli<int> bernoulli_dist(positive_bias);   
  int direction_indicator = oneapi::mkl::rng::device::generate(bernoulli_dist,local_engine);
  int direction = (direction_indicator == 1) ? 1 : -1;

  oneapi::mkl::rng::device::uniform<int> main_heap_distribution(size/device_heap_ptr->blocksize, range);
  int random_step = oneapi::mkl::rng::device::generate(main_heap_distribution,local_engine);

  current_index += (random_step * direction );          

  // Clamping to stay within range
  if (current_index < index_range_start) {
  current_index = index_range_start;

  } else if (current_index >= index_range_end) {
  current_index = index_range_end - 1;

}               

  ret_val = dev_malloc(ptr_superblk , current_index , size, index_range_end, device_heap_ptr->blocksize);
  if(ret_val >= 0)
  {
    alloc_ptr = ret_val +  (current_index *device_heap_ptr->blocksize) + device_heap_ptr->base;
    break;
  }
  if(ret_val <= -1)
  {
    step_size = step_size + 1;
    positive_bias = (positive_bias <= 0.5f) ? positive_bias/0.15f : positive_bias *.2f; 
    range = ((size/device_heap_ptr->blocksize)) ?  (step_size*0.1*(size/device_heap_ptr->blocksize)) + (size/device_heap_ptr->blocksize) : step_size * 1  ;               
  }
  alloc_ptr = 0;

}

  // Debug print in device code
  __spirv_ocl_printf(__malloc_prwalk_debug, prwalk->num_walks,
                     prwalk->walk_length, prwalk->index_range_start,
                     prwalk->index_range_end, prwalk->step_size);
  return reinterpret_cast<void *>(device_heap_ptr->base);
}

DEVICE_EXTERN_C
void free(void *ptr) { return; }


int dev_malloc(unsigned long* device_superblk, unsigned int start_blk, unsigned int size, unsigned int end_blk, unsigned int base_blk_size)
{
  unsigned long ExpectedValue;
  unsigned long DesiredValue;

  int ret_val;  // used to return the offset of the current allocation in the first block assigned into it
  unsigned short iter_count;
  unsigned int num_blks;
  unsigned char shift_count;
  unsigned int  remaining_size;
  unsigned int  allocated_size;
  bool benable_partition_for_size;

   

  if(size == 0)
  {
    return 0;
  }


  iter_count = 0;

  num_blks = 0;
  DesiredValue =0;
  remaining_size = size;
  benable_partition_for_size = false;
  ret_val = -1;

  do
  {

   // sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>atomic_element((device_superblk[start_blk+iter_count]));  
   // ExpectedValue = atomic_element.load();
    ExpectedValue =  __spirv_AtomicLoad(&(device_superblk[start_blk+iter_count]),1, 896 );

    allocated_size =0;
    benable_partition_for_size = false;

    do // this do while loop takes care of failure of compare exchange , we dont need to recalculate the values for the blocks we already determined to be free, 
    {

      if(ExpectedValue & 0xf)// lower nibble
      {
        unsigned char partition;

        partition= (ExpectedValue >> 4) & 0xf ; // upper nibble
        if(partition)
        {   
          unsigned char parition_occupency = (ExpectedValue >> 8) & 0xff;

          if(iter_count == 0) // first block of the allocation 
          { 
            unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);
            unsigned int required_blks = (remaining_size%partition_size) ?   (remaining_size/partition_size) + 1 : (remaining_size/partition_size); //if remaining_size is zero, required blks will be zero and so will be mask. 
            unsigned char mask_position = __spirv_ocl_ctz(~(parition_occupency));
            unsigned int mask = (1<<(required_blks)) - 1;
            bool found_fit = false;
            DesiredValue = 0;

            while((mask_position + (required_blks - 1)) < NUM_PARTITIONS_IN_BLOCK) //needs to be split across blocks when we dont find a fit in the current block
            {
              found_fit = ((parition_occupency & (mask << mask_position)) == 0);
              if(found_fit == false)
              {
                mask_position++;
              }
              else
              break;
            }

            unsigned char count_free_blks = (found_fit) ? required_blks : __spirv_ocl_clz(parition_occupency);
            if(count_free_blks == 0)
            {
              allocated_size = 0;
              ret_val = -1;  //if some one else snatched the block
              break;
            }

            mask = (found_fit) ? mask : (1 << count_free_blks) -1; //recalculate mask when we can accomodate partial allocation in the current block //BUG fix to swap the true and false statements, if found_fit is false then recalculate mask 
            mask = (found_fit) ? (mask << mask_position ) & 0xff : (mask << (NUM_PARTITIONS_IN_BLOCK-count_free_blks)) & 0xff;
            unsigned long alloc_length = (found_fit) ? (1 << ((required_blks<<1) -1)): 0x3;
            unsigned int alloc_mask = ((alloc_length << (mask_position<<1)) | ((ExpectedValue >> 16) & 0xffff));
            DesiredValue  = (found_fit) ? 0 | (ExpectedValue >> 32): size; // BUG FIX. when we can accomdoate complete allocation with in the current block, i.e found_Fit == true, we should retain the value of the size from the expected value in the desired value, as there is no update for the current allocation on the size field
            DesiredValue = (found_fit) ? (alloc_mask << 16) : ((DesiredValue << 32) | (alloc_mask << 16)); // size cannot exceed 4GB
            mask = (mask | parition_occupency);
            DesiredValue = DesiredValue | ( mask << 8 ) | (ExpectedValue & 0xff);
            allocated_size = (found_fit) ? remaining_size : (partition_size * count_free_blks) ;
            ret_val = (found_fit) ? (mask_position * partition_size) : (NUM_PARTITIONS_IN_BLOCK - (__spirv_ocl_clz(parition_occupency)))*partition_size ; //indicates the offset in the block where the allocation is starting , used to calculate the allocation pointer

          }
         else /* for any other block in the middle or last , the allocated blocks must start at the beginning of the block for contiguity. 
              Middle blocks cannot be allocated in an already partitioned blk unless all partitions are empty in which case they need not be partitioned at all.
              so this else shold only tackle last block */
          {
            unsigned char count_free_blks = __spirv_ocl_ctz(parition_occupency) ;
            unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);
            if(remaining_size > (partition_size * count_free_blks))
            {
              allocated_size = 0;
              break; //allocation failure
            }
            else
            {
              unsigned int required_blks = (remaining_size%partition_size) ?   (remaining_size/partition_size) + 1 : (remaining_size/partition_size);
              allocated_size = remaining_size;
              unsigned int mask =  ((1 << required_blks) - 1) & 0xff ;  
              DesiredValue = ( mask << 8);
              DesiredValue = DesiredValue | ExpectedValue ;
              /* NO need to assign ret_val as this is not the first block if current block was partitioned.ret_val should have bee set by the first block allocated */
            }
          }
        }
        else
        {
          allocated_size = 0;
          if(iter_count == 0)
          ret_val = -1;  //whether this is the case for first block only because someone else snatched the allocations, preserve the value of ret_val otherwse to unwind if at all
          break;
        }

      }
      else
      {
        if((remaining_size < base_blk_size) && 
        ((base_blk_size - (remaining_size % base_blk_size)) > ((FRAGMENTATION_MARGIN_IN_PERCENTAGE/100) * base_blk_size)))//look to partition if only blk or the last blk. 
        {
          /* filling partition information here */
          unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);
          unsigned int required_blks =  (remaining_size%partition_size) ?   (remaining_size/partition_size) + 1 : (remaining_size/partition_size); //range 1- 8
          unsigned int mask =  ((1 << required_blks) - 1)  & 0xff ; // the free blks should be in the last for the first iteration of the allocation 
          unsigned int alloc_length = (iter_count == 0) ? (1 << ((required_blks<<1) - 1)) : 0; // this is the first and the last block. we dont need length encoding for last block
          DesiredValue = alloc_length << 16;
          DesiredValue = DesiredValue | ( mask << 8 ) | (((NUM_PARTITIONS_IN_BLOCK <<4)| 0x1) & 0xff);
          allocated_size = remaining_size;
          if(iter_count == 0) //retain the ret_val from the first allocated block when not the first block of the allocation
          ret_val = 0;
        }
        else // this is for all blocks of a given allocation
        {
          DesiredValue = (iter_count == 0) ? (size ) : 0; //allocate the block without partitioning
          DesiredValue = (DesiredValue << 32)| 0x01; //allocate the block without partitioning
          allocated_size = (remaining_size < base_blk_size) ? remaining_size : base_blk_size;
          if(iter_count == 0) //retain the ret_val from the first allocated block when not the first block of the allocation
          ret_val = 0;
        }                
      }

    } while((__spirv_AtomicCompareExchange(&(device_superblk[start_blk+iter_count]),  1, 896, 896, DesiredValue, ExpectedValue) == false));
 

    num_blks = (allocated_size > 0) ? (num_blks + 1) : num_blks;
    remaining_size =  (allocated_size > 0) ? remaining_size - allocated_size : remaining_size ;
    iter_count = (allocated_size > 0) ? iter_count + 1:  iter_count;

  } while((remaining_size > 0) && ((start_blk+iter_count) < ((NUM_OF_SUPERBLOCKS_PER_HEAP* NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK)-1)) && (allocated_size !=0 )); //get all the consequitive superblocks upto count_of_superblks starting at superblk_index

#ifdef INCLUDE_UNWINDING
  /* Since we are doing contiguous block allocation and they can span over multiple super blocks we may not get all the blocksto complete the allocation. In such a case we will unwind to release the blocks */
  unsigned int count_of_blks = 0;
  int deallocate_size = size - remaining_size;
  unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);
  while((remaining_size > 0) && (num_blks > 0)) //loop to unwind and release the blocks
  {
    //sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>atomic_element((device_superblk[start_blk+count_of_blks]));  
   // ExpectedValue = atomic_element.load();
   ExpectedValue =  __spirv_AtomicLoad(&(device_superblk[start_blk+count_of_blks]),1,896);


    do // this do while loop takes care of failure of compare exchange , we dont need to recalculate the values for the blocks we already determined to be free, 
    {
      if((ExpectedValue & 0xf) == 0)
      {
         break; //unwinding error
      }
      else
      {
        if(((ExpectedValue >> 4) & 0xf) != 0)
        {
          unsigned char parition_occupency = (ExpectedValue >> 8) & 0xff;

          if(count_of_blks == 0)
          {
            unsigned int count_allocated_blks = (NUM_PARTITIONS_IN_BLOCK - (ret_val/partition_size));
            unsigned int position = ret_val/partition_size ;
            unsigned int mask = ((1 << count_allocated_blks) -1) << position;
            deallocate_size =  deallocate_size - (partition_size*count_allocated_blks);
            DesiredValue = (~(0x3 << (position<<1)))& 0xffff ;
            DesiredValue = DesiredValue & ((ExpectedValue >> 16) & 0xffff);
            mask = ~mask & ((ExpectedValue >> 8) & 0xff);
            unsigned char partition_info = (mask == 0) ? 0 :  (ExpectedValue & 0xff);
            DesiredValue = (partition_info == 0) ? 0 : (DesiredValue << 16) | (mask << 8) | partition_info;
          }
          else /* last blk ..middle blks will never be partitioned, if found partitioned its a BUG!!! */
          {
            unsigned int count_allocated_blks = (deallocate_size%partition_size) ? (deallocate_size/partition_size) + 1: (deallocate_size/partition_size);
            deallocate_size =  0;
            unsigned int mask = ((1 << count_allocated_blks) -1);
            mask = ~mask & ((ExpectedValue >> 8) & 0xff);
            unsigned char partition_info = (mask == 0) ? 0 :  (ExpectedValue & 0xff);
            DesiredValue = (partition_info == 0) ? 0 : (ExpectedValue) | (mask << 8);
          }
        }
        else
        {
          DesiredValue = 0x00;
          deallocate_size = (deallocate_size >= base_blk_size) ? deallocate_size - base_blk_size : 0;
        }
      }
    } while((__spirv_AtomicCompareExchange(&(device_superblk[start_blk+count_of_blks]),1,896,896, DesiredValue , ExpectedValue) == false));
    num_blks--; //should reach zero to end the loop
    count_of_blks++;
    ret_val = -count_of_blks; // to indicate that we have unwound the allocation 
  }
#endif
  return ret_val;
}
#endif