#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>

#include <c10/util/intrusive_ptr.h> // similar to shared_ptr
#include <c10/util/Exception.h>

namespace c10{

// TODO: LEARN: a struct can inherit from a class, but privately by default.
// final means tis struct cannot be inherited.
struct C10_API StorageImpl final : public c10::intrusive_ptr_target{
private:
  DataPtr data_ptr_;
  size_t size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  // TODO: FIGURE: what does received_cuda_ mean?
  bool received_cuda_;
  Allocator* allocator_;
public:
  // TODO: FIGURE: why to use a empty struct here?
  struct use_byte_size_t {};
  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator) {
    if (resizable){
      TORCH_INTERNAL_ASSERT(allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            allocator->allocate(size_bytes),  // allocate a DataPtr
            allocator,
            resizable) {}

  StorageImpl& operator=(StorageImpl&& other) = default;
  // TODO: NOTE: no implicity copy is allowed.
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) = default;
  // TODO: NOTE: no implicity copy is allowed
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() = default;

  template<typename T>
  inline T* data() const{
    return static_cast<T*>(this->data_ptr_.get());
  }

  template<typename T>
  inline T* unsafe_data() const{
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override{
    // keep device info. set data_ptr_.ptr_.data and data_ptr.ptr_.ctx to nullptr
    this->data_ptr_.clear();
  }

  // TODO: remove later
  void set_nbytes(size_t nbytes){
    size_bytes_ = nbytes;
  }

  size_t nbytes() const{
    return size_bytes_;
  }

  bool resizable() const {
    return resizable_;
  };

  c10::DataPtr& data_ptr(){
    return data_ptr_;
  }

  const c10::DataPtr& data_ptr() const{
    return data_ptr_;
  }

  c10::DataPtr set_data_ptr(c10::DataPtr&& data_ptr){
    // data_ptr is a right-side ref
    std::swap(data_ptr_, data_ptr);  // change data_ptr to left-side ref
    return std::move(data_ptr);
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    return data_ptr_.get();
  }

  void* data() const {
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    return allocator_;
  }

  const at::Allocator* allocator() const {
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  Device device() const {
    return data_ptr_.device();
  }

  void set_resizable(bool resizable) {
    if (resizable) {
      // We need an allocator so that StorageImpl can be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   * This method is try to re-set the data that the StorageImpl manages.
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
      // TODO: NOTE: here we can know that the ctx in DataPtr holds the data pointer itself
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;
    resizable_ = false;
  }

/*
  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }
*/
};
}