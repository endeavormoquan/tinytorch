#pragma once

// for ostream, to_string
#include <iostream>
#include <string>
#include <c10/macros/Macros.h>

namespace c10 {

// Semantically, a dispatch key identifies a possible "level" in our
// dispatch, for which a handler may be registered.  Traditional
// backends like CPU and CUDA get dispatch keys; however, so do
// "wrapping" layers like Variable (for autograd handling).
// TODO: FIGURE: how to understand dispatch keys other than CPU/CUDA?
// In implementation terms, the dispatch key identifies a specific "bit" in a
// DispatchKeySet.  Higher bit indexes get handled by dispatching first (because
// we "count leading zeros" when we extract the highest priority dispatch
// key.)
enum class DispatchKey : uint8_t {

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ UNDEFINED ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // This is not a "real" tensor id, but it exists to give us a "nullopt"
  // element we can return for cases when a DispatchKeySet contains no elements.
  // You can think a more semantically accurate definition of DispatchKey is:
  //
  //    using DispatchKey = optional<RealDispatchKey>
  //
  // and Undefined == nullopt.  We didn't actually represent
  // it this way because optional<RealDispatchKey> would take two
  // words, when DispatchKey fits in eight bits.

  Undefined = 0,

  // Define an alias for Undefined to represent CatchAll (long term
  // this will get eliminated, but for now it's convenient)
  CatchAll = Undefined,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~ BACKENDS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // A "backend" is colloquially used to refer to handlers for dispatch
  // which actually implement the numerics of an operation in question.
  //
  // Due to the nature of the enum, these backends are specified in
  // an ordered way, but for most backends this order is not semantically
  // meaningful (e.g., it's valid to reorder these backends without changing
  // semantics).  The only situation when backend ordering is meaningful
  // is when the backend participates in multiple dispatch with another
  // backend; e.g., CPU and SparseCPU (sparse must have
  // higher priority).

  // Here are backends which you think of as traditionally specifying
  // how to implement operations on some device.
  CPU, // registered at build/aten/src/ATen/CPUType.cpp
  SparseCPU, // registered at build/aten/src/ATen/SparseCPUType.cpp
  FPGA, // FIGURE: where to register FPGA? see code context for more info.
  // The meta function characterizes how an operation affects the metadata of a
  // tensor (shape, dtype) without doing any of the actual computation.  A
  // meta tensor can be used to dry run operators without actually doing
  // any computation, e.g., add on two meta tensors would give you another
  // meta tensor with the output shape and dtype, but wouldn't actually
  // add anything.  A meta implementation typically would look something like:
  //
  //  Tensor meta::add(const Tensor& self, const Tensor& other) {
  //    TORCH_CHECK(self.size().equals(other.size()));
  //    return at::empty_like(self, self.size());
  //  }
  //
  // The meta function would get invoked if you ran an operator passing
  // in meta tensors.  The call stack in such a case would look something like
  // this:
  //
  //  at::add(x: Meta, y: Meta) {
  //    return [dispatch] meta::add(x: Meta, y: Meta) {
  //      output_shape = ...
  //      [dispatch] meta::empty(output_shape) {
  //        return ... meta tensor with output_shape but no data allocated ...
  //      }
  //    }
  //  }
  //
  // Meta functions have an important secondary function, which is they can
  // be used as tensor "allocators".  A typical backend implementation should
  // be implemented in this way:
  //
  //  Tensor cpu::add(const Tensor& self, const Tensor& other) {
  //    Tensor result = meta::add(self, other);
  //    // ... do the actual computation into result ...
  //    return result;
  //  }
  //
  // In this case, the internal at::empty_like invocation would dispatch to the
  // CPU factory function, not the meta factory function.  The call stack in
  // this case looks like:
  //
  //  at::add(x: CPU, y: CPU) {
  //    return [dispatch] cpu::add(x: CPU, y: CPU) {
  //      output = [direct] meta::add(x: CPU, y: CPU) {
  //        output_shape = ...
  //        [dispatch] cpu::empty(output_shape)
  //      }
  //      ... compute on output ...
  //      return output;
  //    }
  //  }
  //
  Meta,

  // In some situations, it is not immediately obvious what the correct
  // backend for function is, because the function in question doesn't
  // have any "tensor" arguments.  In this case, a BackendSelect function
  // can be registered to implement the custom determination of the
  // correct backend.
  BackendSelect,

  // The named dispatch key is set for any tensors with named dimensions.
  // Although we have a dispatch key for named tensors, for historical reasons,
  // this dispatch key doesn't do any of the substantive functionality for named
  // tensor (though, hypothetically, it could!)  At the moment, it's just
  // responsible for letting us give good error messages when operations
  // don't support named tensors.
  //
  // NB: If you ever consider moving named tensor functionality into
  // this dispatch key, note that it might be necessary add another dispatch
  // key that triggers before composite operators, in case a composite operator
  // has named dimension propagation that doesn't match that of its
  // constituent parts.
  Named,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ AUTOGRAD ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // All backends are oblivious to autograd; autograd is handled as a
  // layer which happens on top of all backends.  It inspects the autograd
  // metadata of all inputs, determines what autograd metadata should be
  // constructed by the output, and otherwise defers to the backend to
  // actually do the numeric computation.  Autograd contains
  // the bulk of this logic.
  Autograd,

  Profiler,

  Tracer,

  // Autocasting precedes VariableTypeId, to ensure casts are autograd-exposed
  // and inputs are saved for backward in the post-autocast type.
  Autocast,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WRAPPERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // There are a number of alternative modes which may want to handle before
  // autograd; for example, error checking, tracing, profiling or vmap.  They
  // go here.

  // This is the dispatch key for BatchedTensorImpl, which is used to implement
  // batching rules for vmap.
  Batched,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a single
  // process test.  Use it by creating a TensorImpl with this DispatchKey, and
  // then registering operators to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp for a usage example.
  TESTING_ONLY_GenericWrapper,

  // TESTING: This is intended to be a generic testing tensor type id.
  // Don't use it for anything real; its only acceptable use is within a ingle
  // process test.  Use it by toggling the mode on and off via
  // TESTING_ONLY_tls_generic_mode_set_enabled and then registering operators
  // to operate on this type id.  See
  // aten/src/ATen/core/dispatch/backend_fallback_test.cpp
  // for a usage example
  TESTING_ONLY_GenericMode,

  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  NumDispatchKeys, // Sentinel

  // ~~~~~~~~~~~~~~~~~~~~~~~~~ BC ALIASES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
  // The aliases exist for backwards compatibility reasons, they shouldn't
  // be used
  CPUTensorId = CPU,
};

// Note [Private use DispatchKey]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Private use tensor IDs are preallocated tensor type IDs for use in user
// applications.  Similar to private use fields in HTTP, they can be used
// by end users for experimental or private applications, without needing
// to "standardize" the tensor ID (which would be done by submitting a PR
// to PyTorch to add your type ID).
//
// Private use tensor IDs are appropriate to use if you want to experiment
// with adding a new tensor type (without having to patch PyTorch first) or
// have a private, non-distributed application that needs to make use of a
// new tensor type.  Private use tensor IDs are NOT appropriate to use for
// libraries intended to be distributed to further users: please contact
// the PyTorch developers to get a type ID registered in this case.
//
// We provide two classes of private user tensor id: regular DispatchKeys
// and PreAutograd DispatchKeys.  DispatchKeys serve the role of ordinary "backend"
// DispatchKeys; if you were adding support for a new type of accelerator, you
// would use a DispatchKey, and reuse autograd definitions already defined in
// PyTorch for operators you define.  PreAutograd DispatchKeys serve as "wrapper"
// DispatchKeys: they are most appropriate for tensors that compose multiple
// internal tensors, and for cases when the built-in autograd formulas for
// operators are not appropriate.

static_assert(
  static_cast<uint8_t>(DispatchKey::NumDispatchKeys) < 64,
  "DispatchKey is used as index into 64-bit bitmask; you must have less than 64 entries");

C10_API const char* toString(DispatchKey);
C10_API std::ostream& operator<<(std::ostream&, DispatchKey);

// These are some convenience identifiers for dispatch keys which are
// shorter to type than their long counterparts.  Note that some of these
// dispatch keys directly correspond to DeviceType; and most APIs that
// accept DispatchKey also accept DeviceType; e.g.,
// torch::dispatch(torch::kCPU, ...) is also valid.
constexpr DispatchKey kAutograd = DispatchKey::Autograd;

} // namespace c10

namespace torch {
  // Expose the constant, but not the TYPE (DispatchKey is an implementation
  // detail!)
  using c10::kAutograd;
}

// NB: You really shouldn't use this instance; this enum is guaranteed
// to be pretty small so a regular array should be acceptable.
// TODO: NOTE: actually in Allocator.h, we just use an array to store info of different Keys.
namespace std {
template <>
struct hash<c10::DispatchKey> {
  typedef size_t result_type;
  typedef c10::DispatchKey argument_type;

  size_t operator()(c10::DispatchKey x) const {
    return static_cast<size_t>(x);
  }
};
}
