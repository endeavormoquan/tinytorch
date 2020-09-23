#include <c10/core/Allocator.h>

#include <c10/util/Exception.h>

namespace c10{
// TODO: NOTE: we can not define makeDataPtr in header file (or it is not easy)
// here we define a ctx deleter
static void deleteInefficientStdFunctionContext(void* ptr){
  delete static_cast<InefficientStdFunctionContext*>(ptr); // delete a ctx
}

c10::DataPtr InefficientStdFunctionContext::makeDataPtr(
    void* ptr,
    const std::function<void(void*)>& deleter,
    Device device){
      return DataPtr(/*data*/ ptr,
                     /*ctx*/new InefficientStdFunctionContext({ptr, deleter}),  // ctx contains a pointer to data and a deleter function
                     /*ctx deleter*/&deleteInefficientStdFunctionContext,
                     device
                     );
}

// TODO: NOTE: since each DeviceType has a specific allocator, so we define a array to represent them
C10_API c10::Allocator* allocator_array[c10::COMPILE_TIME_MAX_DEVICE_TYPES]; // defined in DeviceType.h
C10_API uint8_t allocator_priority[c10::COMPILE_TIME_MAX_DEVICE_TYPES] = {0}; // bigger value, higher priority

c10::Allocator* GetAllocator(const c10::DeviceType& t){
  auto* allocator = allocator_array[static_cast<int>(t)];  // since DeviceType is enum, so we can cast it to int
  // should check if allocator for devicetype t has been registered
  TORCH_INTERNAL_ASSERT(allocator, "Allocator for ", t, " is not exist.");
  return allocator;
}

void SetAllocator(c10::DeviceType& t, c10::Allocator* allocator, uint8_t priority){
  // TODO: NOTE: here is a rule, we can only replace original allocator when a new one has a higner priority
  auto DeviceType2Int = static_cast<int>(t);
  if (priority >= allocator_priority[DeviceType2Int]){
    allocator_array[DeviceType2Int] = allocator;
    allocator_priority[DeviceType2Int] = priority;
  }
  else{
    // do nothing.
  }
}
}