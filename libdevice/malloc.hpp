#pragma once
#include "spirv_vars.h"

#include "onemkl_philox4x32x10_uniform_bernoulli.hpp"

#define HEAP_SIZE                                  (0x800000)
#define NUM_OF_HEAPS                               (1)

#define NUM_OF_SUPERBLOCKS_PER_HEAP                (2048)
#define NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK          (32)
#define INCLUDE_UNWINDING                          (1)
#define HEAP_10_PERCENT_PARTITION                  (NUM_OF_SUPERBLOCKS_PER_HEAP/10)  // upto 10%
#define HEAP_20_PERCENT_PARTITION                  (HEAP_10_PERCENT_PARTITION * 2)  // upto 20%
#define NUM_PARTITIONS_IN_BLOCK                    (8)
#define FRAGMENTATION_MARGIN_IN_PERCENTAGE         (10) 

// super block structure
union superblk {
  unsigned long  SuperblkMetadata;
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

  unsigned long  base_seed;
  unsigned long  base_subseed;
};

struct device_heap_t {
  unsigned long  base;
  unsigned long  size;
  unsigned int blocksize;
  unsigned int max_num_blocks;

  unsigned int max_num_heap_partitions;
  unsigned long  *ptr_superblk;
  struct random_walk_params_t *prwalkparams;
};

struct allocator_context_t
{
	struct device_heap_t* deviceheap[NUM_OF_HEAPS];
};


int dev_malloc(unsigned long * device_superblk, unsigned int start_blk, unsigned int size, unsigned int end_blk, unsigned int base_blk_size);
unsigned int dev_free(unsigned long* device_superblk, unsigned int start_blk, unsigned int byte_offset, unsigned int base_blk_size);

DeviceGlobal<void *> __DeviceAllocCtxPtr;

#if defined(__SPIR__) || defined(__SPIRV__)

#define __SYCL_CONSTANT__ __attribute__((opencl_constant))
#define SPIR_GLOBAL __attribute__((opencl_global))

#ifdef INCLUDE_SPIRV_OCL_PRINT
static const __SYCL_CONSTANT__ char __malloc_prwalk_debug[] =
    "[kernel] random walk params fileds: num_walks=%d, walk_length=%d, "
    "index_range_start=%d, index_range_end=%d, step_size=%d\n";


static const __SYCL_CONSTANT__ char __malloc_prwalk_debug1[] =
    "hello world: local_id=%d\n";
static const __SYCL_CONSTANT__ char __malloc_prwalk_debug2[] =
    "hello world: ADDR=%lx\n";

static const __SYCL_CONSTANT__ char __malloc_prwalk_debug3[] =
    "hello world: ADDR=%lx\n";

static const __SYCL_CONSTANT__ char __malloc_prwalk_debug4[] =
    "Free Success: ADDR=%lx\n";

static const __SYCL_CONSTANT__ char __malloc_prwalk_debug5[] =
    "Free Failed: ADDR=%lx\n";
#endif

#ifdef INCLUDE_SPIRV_OCL_PRINT
extern SYCL_EXTERNAL int
__spirv_ocl_printf(const __SYCL_CONSTANT__ char *Format, ...);
#endif

extern DEVICE_EXTERNAL int __spirv_ocl_ctz(int) noexcept ;
//extern DEVICE_EXTERNAL int __spirv_ocl_clz(unsigned char) noexcept;
extern DEVICE_EXTERNAL int __spirv_ocl_ctz(unsigned char) noexcept;

extern DEVICE_EXTERNAL unsigned long __spirv_AtomicCompareExchange(unsigned long SPIR_GLOBAL*, int,
                                                         int, int, unsigned long ,
                                                         unsigned long ) noexcept;

 //extern SYCL_EXTERNAL int __spirv_AtomicCompareExchange(unsigned long *, int,
//int, int, unsigned long ,
//unsigned long ) noexcept;

extern DEVICE_EXTERNAL unsigned long __spirv_AtomicLoad(unsigned long SPIR_GLOBAL*, int,
int) noexcept;

//extern DEVICE_EXTERNAL unsigned long  __spirv_AtomicLoad(SPIR_GLOBAL const unsigned long *, int,
//                                              int) noexcept;
DEVICE_EXTERN_C

void *malloc(size_t size) {

    unsigned long  alloc_ptr = 0;
  struct allocator_context_t *temp = reinterpret_cast<struct allocator_context_t *> (__DeviceAllocCtxPtr.get());

#ifdef INCLUDE_SPIRV_OCL_PRINT
   __spirv_ocl_printf(__malloc_prwalk_debug2, reinterpret_cast<unsigned long>(temp));
#endif

  struct device_heap_t *device_heap_ptr = reinterpret_cast<struct device_heap_t *>(temp->deviceheap[0]);

 #ifdef INCLUDE_SPIRV_OCL_PRINT
    __spirv_ocl_printf(__malloc_prwalk_debug3, device_heap_ptr);
#endif

  random_walk_params_t *prwalk = device_heap_ptr->prwalkparams;

  unsigned long * ptr_superblk = device_heap_ptr->ptr_superblk;

#ifdef  INCLUDE_SPIRV_OCL_PRINT
     __spirv_ocl_printf(__malloc_prwalk_debug2, reinterpret_cast<unsigned long>(prwalk));



    __spirv_ocl_printf(__malloc_prwalk_debug3, ptr_superblk);
#endif

  //id<1> global_id =item.get_global_id(); //need equivalent of this in the compiler..
  int walklength = prwalk->walk_length;
  int walk_id = __spirv_BuiltInGlobalLinearId() ; // NHOW TO GET GOLBAL ID FOR WORKITEM ..?global_id; 
  float positive_bias = 0.5f;

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

    unsigned int local_id = (__spirv_BuiltInWorkgroupSize(1) * __spirv_BuiltInWorkgroupSize(0) *    \
                            __spirv_BuiltInLocalInvocationId(2)) +                                 \
                            (__spirv_BuiltInWorkgroupSize(0) *                                      \
                            __spirv_BuiltInLocalInvocationId(1)) +                                 \
                            __spirv_BuiltInLocalInvocationId(0);
  // Debug print in device code
#ifdef INCLUDE_SPIRV_OCL_PRINT
   __spirv_ocl_printf(__malloc_prwalk_debug1,local_id);

 // __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInLocalInvocationId(0));
  //  __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInLocalInvocationId(1));
  //    __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInWorkgroupSize(0));
   //     __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInLocalInvocationId(2));
   //       __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInWorkgroupSize(0));
   //         __spirv_ocl_printf(__malloc_prwalk_debug1, __spirv_BuiltInWorkgroupSize(1));
#endif    

    int block_range = (modulo_size == 1)? device_heap_ptr->max_num_blocks : (partition_range * NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK);
    index_range_start = (modulo_size == 1) ? 0 : (local_id % modulo_size)*(partition_range * NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK);
    index_range_end = index_range_start + (block_range -1);

    num_heap_blks = (size / device_heap_ptr->blocksize) + 1; // this is the minimum numer of bigger sized blocks required to perform subheap allocation as the sub heap allocation may or may not start on a main block boundary

  }

#ifdef INCLUDE_SPIRV_OCL_PRINT
 __spirv_ocl_printf(__malloc_prwalk_debug1,size);
#endif
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
#ifdef INCLUDE_SPIRV_OCL_PRINT        
__spirv_ocl_printf(__malloc_prwalk_debug3,(ptr_superblk));
__spirv_ocl_printf(__malloc_prwalk_debug1,(current_index));
__spirv_ocl_printf(__malloc_prwalk_debug1,(size));
__spirv_ocl_printf(__malloc_prwalk_debug1,(index_range_end));
__spirv_ocl_printf(__malloc_prwalk_debug1,(device_heap_ptr->blocksize));
#endif

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

#ifdef INCLUDE_SPIRV_OCL_PRINT   
  // Debug print in device code
  __spirv_ocl_printf(__malloc_prwalk_debug, prwalk->num_walks,
                     prwalk->walk_length, prwalk->index_range_start,
                     prwalk->index_range_end, prwalk->step_size);
#endif

  return reinterpret_cast<void *>(alloc_ptr);
}

DEVICE_EXTERN_C
void free(void *ptr) {

  /* Invoke kernel for freeing */
  if(ptr != 0) //for valid allocation pointers
  {
    unsigned int byte_offset;
    unsigned short int superblk_offset;
    unsigned int block_offset;
    unsigned int partition_bytes;
    unsigned int ret_val;
    unsigned long alloc_ptr = reinterpret_cast<unsigned long>(ptr);

    partition_bytes = 0;

    struct allocator_context_t *temp = reinterpret_cast<struct allocator_context_t *> (__DeviceAllocCtxPtr.get());


    struct device_heap_t *device_heap_ptr = reinterpret_cast<struct device_heap_t *>(temp->deviceheap[0]);

#ifdef INCLUDE_SPIRV_OCL_PRINT
   __spirv_ocl_printf(__malloc_prwalk_debug2, reinterpret_cast<unsigned long>(temp));
    __spirv_ocl_printf(__malloc_prwalk_debug3, device_heap_ptr);
#endif
    unsigned long * ptr_superblk = device_heap_ptr->ptr_superblk;


    byte_offset = alloc_ptr - device_heap_ptr->base;

    block_offset = (byte_offset / device_heap_ptr->blocksize); //which block 
    partition_bytes = byte_offset % (device_heap_ptr->blocksize);

    ret_val = dev_free(ptr_superblk,block_offset,partition_bytes,device_heap_ptr->blocksize);

    if(ret_val == 0)
    {
#ifdef INCLUDE_SPIRV_OCL_PRINT
      __spirv_ocl_printf(__malloc_prwalk_debug4, ptr);
#endif
    }
    else
    {
#ifdef INCLUDE_SPIRV_OCL_PRINT
      __spirv_ocl_printf(__malloc_prwalk_debug5, ptr);
#endif
    }

  }


  return; 
}


int dev_malloc(unsigned long * device_superblk, unsigned int start_blk, unsigned int size, unsigned int end_blk, unsigned int base_blk_size)
{
  unsigned long  ExpectedValue;
  unsigned long  DesiredValue;

  int ret_val;  // used to return the offset of the current allocation in the first block assigned into it
  unsigned short iter_count;
  unsigned int num_blks;
  unsigned char shift_count;
  unsigned int  remaining_size;
  unsigned int  allocated_size;
  bool benable_partition_for_size;
  bool bloop_back;
  unsigned long atomic_exgh_value;
   

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
  bloop_back = false;
  do
  {

   // sycl::atomic_ref<unsigned long , sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>atomic_element((device_superblk[start_blk+iter_count]));  
   // ExpectedValue = atomic_element.load();
    ExpectedValue =  __spirv_AtomicLoad((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk+iter_count]),1, 896 );

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
            unsigned long  alloc_length = (found_fit) ? (1 << ((required_blks<<1) -1)): 0x3;
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
      bloop_back = false;
      atomic_exgh_value = __spirv_AtomicCompareExchange((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk+iter_count]),  1, 896, 896, (unsigned long) DesiredValue, (unsigned long)ExpectedValue);
      if(atomic_exgh_value != ExpectedValue) //check if atomic exchange failed
      {
        ExpectedValue = atomic_exgh_value;
        bloop_back = true;
      }
    } while(bloop_back == true);
 

    num_blks = (allocated_size > 0) ? (num_blks + 1) : num_blks;
    remaining_size =  (allocated_size > 0) ? remaining_size - allocated_size : remaining_size ;
    iter_count = (allocated_size > 0) ? iter_count + 1:  iter_count;

  } while((remaining_size > 0) && ((start_blk+iter_count) < ((NUM_OF_SUPERBLOCKS_PER_HEAP* NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK)-1)) && (allocated_size !=0 )); //get all the consequitive superblocks upto count_of_superblks starting at superblk_index

#ifdef INCLUDE_UNWINDING
  /* Since we are doing contiguous block allocation and they can span over multiple super blocks we may not get all the blocksto complete the allocation. In such a case we will unwind to release the blocks */
  unsigned int count_of_blks = 0;
  int deallocate_size = size - remaining_size;
  unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);
  bloop_back = false;
  while((remaining_size > 0) && (num_blks > 0)) //loop to unwind and release the blocks
  {
    //sycl::atomic_ref<unsigned long , sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>atomic_element((device_superblk[start_blk+count_of_blks]));  
   // ExpectedValue = atomic_element.load();
   ExpectedValue =  __spirv_AtomicLoad((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk+count_of_blks]),1,896);


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
      
      bloop_back = false;      
      atomic_exgh_value =__spirv_AtomicCompareExchange((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk+count_of_blks]),1,896,896, (unsigned long)DesiredValue , (unsigned long)ExpectedValue);
      if(atomic_exgh_value != ExpectedValue) //check if atomic exchange failed
      {
        ExpectedValue = atomic_exgh_value;
        bloop_back = true;
      }
    } while(bloop_back == true);

    num_blks--; //should reach zero to end the loop
    count_of_blks++;
    ret_val = -count_of_blks; // to indicate that we have unwound the allocation 
  }
#endif
  return ret_val;
}


unsigned int dev_free(unsigned long* device_superblk, unsigned int start_blk, unsigned int byte_offset, unsigned int base_blk_size)
{
    unsigned long ExpectedValue;
    unsigned long DesiredValue;
    unsigned int ret_val;
    unsigned int size;
    unsigned int deallocated_size;
    unsigned int iter_count = 0;    
    unsigned long atomic_exgh_value;
    bool bloop_back;
    

    ret_val = 0;
    DesiredValue = 0;
	iter_count  = 0;

	size = 0;
	deallocated_size = 0;
  bloop_back = false;
    do
    {
		//sycl::atomic_ref<unsigned long, sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::access::address_space::global_space>atomic_element((device_superblk[start_blk + iter_count]));  
		//ExpectedValue = atomic_element.load();
      ExpectedValue =  __spirv_AtomicLoad((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk + iter_count]),1, 896 );

        do
        {
            deallocated_size = 0; // need to reset to take care of compare exchange failure that could lead to updated expected value
            if(ExpectedValue & 0xf)// lower nibble byte0
            {
                unsigned char partition;
                unsigned short desired_byte1;
                unsigned int desired_byte4567;

                partition= (ExpectedValue >> 4) & 0xf ; // upper nibble byte0
                if(partition)
                {   
                  unsigned int blk_bit_vector = (ExpectedValue >> 8) & 0xff;
                  unsigned short blk_length_vector = (ExpectedValue >> 16) & 0xffff;
                  unsigned int ExAlloc_length =  (ExpectedValue >> 32) & 0xffffffff; 
                  unsigned int partition_size = (base_blk_size/NUM_PARTITIONS_IN_BLOCK);

                    if(iter_count == 0) // first block of the allocation being freed
                    {

                      unsigned int blk_index =  (byte_offset/partition_size) ;  //must be with in 0 through  6 , as 7th index is the last partition within the block

                      /* Its a bug to have allocation length as zero at the first block during freeing. 
                      Possible values are 00 - zero length ,xx --allocation span is with in this block nad the numebr indicates how many blks,
                      11 -allocation cross block boundary so interpret the allocation length from the 16bits*/
                      unsigned char allocation_type = (blk_length_vector >> (blk_index<<1)) & 0x3 ;  //blk_offset range is 0 through  6  and allocation length is 1 - 8.


                      size = (allocation_type == 0x3) ? ExAlloc_length : 0;  // reset to extract cross block allocation length if its a cross block allocation
                      desired_byte4567 = (allocation_type == 0x3) ?  0 : ExAlloc_length ; // this free resets the value to 0 if the allocation is a cross block allocation
             

                      if(allocation_type != 0x3) /* with-in the current block */
                      {
                        unsigned short bit_alloc_len = (blk_length_vector >> (blk_index<<1)) & 0xffff;
                        unsigned short alloc_len     = (__spirv_ocl_ctz(bit_alloc_len) + 1) >> 1;
                        unsigned int bit_vector_mask = ((1 << (alloc_len)) - 1) & 0xff ;
                        bit_vector_mask = bit_vector_mask << blk_index;
                        blk_bit_vector = ((~bit_vector_mask) & blk_bit_vector) & 0xff;
                        unsigned char partition_info = (blk_bit_vector == 0) ? 0 : (ExpectedValue & 0xff);
                        unsigned int alloc_len_mask  = ((1 << (alloc_len <<1)) - 1)& 0xffff; // prepare mask to shut off all the bits for this allocation
                        alloc_len_mask = alloc_len_mask << (blk_index << 1); // BUG FIX ..the allocalength _ mask is 2bits per partition block so blk_index<<1
                        alloc_len_mask = (~(alloc_len_mask) & blk_length_vector) & 0xffff;
                        DesiredValue = (partition_info == 0) ? 0 : desired_byte4567;
                        DesiredValue = (partition_info == 0) ? 0 : (DesiredValue << 32 | (alloc_len_mask << 16) | (blk_bit_vector << 8) | (partition_info));
                        size  = (alloc_len * partition_size);

                        deallocated_size = size;
                      }
                      else
                      {
                        unsigned int bit_vector_mask = ((1 << (NUM_PARTITIONS_IN_BLOCK-blk_index)) - 1) & 0xff ;
                        bit_vector_mask     = ~(bit_vector_mask << blk_index) ;	
                        blk_bit_vector = (bit_vector_mask & blk_bit_vector) & 0xff;
                        unsigned char partition_info = (blk_bit_vector == 0) ? 0 : (ExpectedValue & 0xff);
                        unsigned short alloc_len_mask = (~(0x3 << (blk_index<<1))) & 0xffff;
                        unsigned int desired_alloc_len_mask  = (blk_bit_vector == 0) ?  0: (alloc_len_mask & ((ExpectedValue >> 16) & 0xffff));
                        DesiredValue = 0;
                        DesiredValue = (partition_info == 0) ? 0 : (desired_alloc_len_mask << 16) | (blk_bit_vector << 8) | (partition_info);
                        deallocated_size = (NUM_PARTITIONS_IN_BLOCK-blk_index) * 	partition_size;		
                      }


                    }
                    else /* any other block - as in middle or last..middle blks should never be partitioned as contiguity is lost  */ 
                    {
                      /* The blocks should be at the start of the byte as other wise contiguity is lost  */
                      unsigned int blk_count = (size%partition_size) ? (size/partition_size) + 1 : (size/partition_size); 
                      unsigned int blk_bit_vector_mask =  ((1 << blk_count) - 1) & 0xff ;         

                      blk_bit_vector = ((~(blk_bit_vector_mask)) & blk_bit_vector) & 0xff;
                      unsigned char partition_info = (blk_bit_vector == 0) ? 0 : (ExpectedValue & 0xff);
                      DesiredValue = (partition_info == 0) ? 0 : (blk_bit_vector << 8); 
                      DesiredValue = (partition_info == 0) ? 0 : (DesiredValue) | ExpectedValue;      
                      deallocated_size =	size;	

                    }

                }
                else // not partitioned - first, middle , last
                {
                  size = (iter_count == 0) ? (ExpectedValue >> 32) & 0xffffffff : size;
                  DesiredValue = 0;
                  deallocated_size = base_blk_size;
                }

            }
            else
            {
              size = 0;
              break;
            }
            bloop_back = false;
            atomic_exgh_value = __spirv_AtomicCompareExchange((unsigned long SPIR_GLOBAL*)&(device_superblk[start_blk + iter_count]),1,896,896, (unsigned long)DesiredValue , (unsigned long)ExpectedValue) ;
            if(atomic_exgh_value != ExpectedValue) //check if atomic exchange failed
            {
                ExpectedValue = atomic_exgh_value;
                bloop_back = true;
            }

        }while(bloop_back == true);

          size = (deallocated_size > 0) ? size-deallocated_size : size; // this needs ot be decremented only if the allocation crosses the block
          iter_count = (deallocated_size > 0) ? iter_count+1 : iter_count; // incremented to go to the next block
          ret_val = size;

    } while((size > 0) && ((start_blk+iter_count) < ((NUM_OF_SUPERBLOCKS_PER_HEAP* NUM_OF_HEAP_BLOCKS_PER_SUPERBLOCK)-1))); // the allocation should not cross the number of blocks available

    return ret_val;

}

#endif