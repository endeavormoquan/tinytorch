#pragma once
// TODO: NOTE: Storage is a wrapper of StorageImpl, but I don't know why we should wrap it.
#include <c10/core/StorageImpl.h>

namespace c10{
struct C10_API Storage{
private:
  // TODO: FIGURE: why not just write: StorageImpl storage_impl_; ?
  c10::intrusive_ptr<StorageImpl> storage_impl_; // remember storage_impl_ is a pointer.
  // StorageImpl* storage_impl;
public:
  struct use_byte_size_t {};
  Storage() = delete;  // a storage must be initialized with some data
  Storage(c10::intrusive_ptr<StorageImpl> ptr) : storage_impl_(std::move(ptr)){}
  Storage(Storage&& other) = default;  // should I add this?

  // Allocates memory buffer using given allocator and creates a storage with it
  Storage(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      Allocator* allocator,
      bool resizable)
      : storage_impl_(c10::make_intrusive<StorageImpl>( // use make_xxx to initialize an intrusive ptr
            StorageImpl::use_byte_size_t(),
            size_bytes,
            allocator,  // when construct StorageImpl, it will use allocator->allocate() to allocate data
            resizable)) {}

  // Creates storage with pre-allocated memory buffer. Allocator is given for
  // potential future reallocations, however it can be nullptr if the storage
  // is non-resizable
  Storage(
      use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : storage_impl_(c10::make_intrusive<StorageImpl>(
            StorageImpl::use_byte_size_t(),
            size_bytes,
            std::move(data_ptr), // avoid copy
            allocator,
            resizable)) {}

  // Legacy constructor for partially initialized (dtype or memory) storages
  // that can be temporarily created with Caffe2 APIs. See the note on top of
  // TensorImpl.h for details.
  static Storage create_legacy(const c10::Device device){
    auto allocator = c10::GetAllocator(device.type());
    return Storage(c10::make_intrusive<StorageImpl>(
      StorageImpl::use_byte_size_t(),  // empty struct
      0,  // data size in bytes
      allocator->allocate(0),  // DataPtr
      allocator,  //allocator
      true  // resizable
    ));
  }

  template<typename T>
  T* data() const{
    // TODO: LEARN: how to acess shared_ptr
    return storage_impl_->data<T>();  // (1)
    // equal to: (*storage_impl_).data<T>()  // (2)
    // equal to: storage_impl_.get()->data<T>();  // (3)
    // (2)=(3), because (*struct_pointer).mem == struct_pointer->mem
    // (1)=(2), because of feature of shared_ptr
    // priority of '.' is higher than '*'
  }

  void* data() const{
    return storage_impl_->data(); // without static cast
  }

  template<typename T>
  T* unsafe_data() const{
    return (*storage_impl_).unsafe_data<T>();
  }

  c10::DataPtr& data_ptr() {
    return storage_impl_->data_ptr();
  }

  const c10::DataPtr& data_ptr() const{
    return storage_impl_->data_ptr();
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) const {
    storage_impl_->set_nbytes(size_bytes);
  }

  bool resizable() const {
    return storage_impl_->resizable();
  }

  size_t nbytes() const {
    return storage_impl_->nbytes();
  }

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) const {
    return storage_impl_.get()->set_data_ptr(std::move(data_ptr));
  };

  DeviceType device_type() const {
    return storage_impl_->device_type();
  }

  at::Allocator* allocator() const {
    return storage_impl_.get()->allocator();
  }

  at::Device device() const {
    return storage_impl_->device();
  }

  // TODO: FIGURE: why these methods are unsafe?
  StorageImpl* unsafe_release_storageimpl(){
    return storage_impl_.release(); // get the previous pointer and set storage_impl_ to be nullptr
  }

  StorageImpl* unsafe_get_storageimpl() const{
    return storage_impl_.get();
  }

  operator bool() const {
    return storage_impl_;
  }

  size_t use_count() const {
    return storage_impl_.use_count();
  }

  inline bool unique() const {
    return storage_impl_.unique();
  }

  bool is_alias_of(const Storage& other) const {
    return storage_impl_ == other.storage_impl_;
  }

  void UniqueStorageShareExternalPointer(
      void* src,
      size_t capacity,
      DeleterFnPtr d = nullptr) {
    if (!storage_impl_.unique()) {
      TORCH_INTERNAL_ASSERT(false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(src, capacity, d);
  }

  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t capacity) {
    if (!storage_impl_.unique()) {
      TORCH_INTERNAL_ASSERT(false,
          "UniqueStorageShareExternalPointer can only be called when use_count == 1");
    }
    storage_impl_->UniqueStorageShareExternalPointer(
        std::move(data_ptr), capacity);
  }
};
}