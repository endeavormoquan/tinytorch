#pragma once

#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>
#include <c10/util/llvmMathExtras.h>
#include <iostream>
#include <string>

namespace c10{

class DispatchKeySet final{
private:
  uint64_t repr_ = 0; // a bit represents a Key. Higher bit, higher priority
  // we indeed have a constructor that receive an uint64_t value, but it MUST with RAW
  DispatchKeySet(uint64_t x):repr_(x) {} // this constructor is use only internal, and the usage applys C++11 features.
public:
  // TODO: LEARN: use one element's enum to represent different args
  enum Full {FULL};
  enum FullAfter {FULL_AFTER};
  enum Raw {RAW};
  DispatchKeySet()
    : repr_(0) {}
  DispatchKeySet(Full)
    : repr_(std::numeric_limits<decltype(repr_)>::max()) {}
  DispatchKeySet(FullAfter, DispatchKey t)
    // LSB after t are OK, but not t itself.
    // TODO: NOTE: take SparseCPU as an example, SparseCPU is actually 2U,
    // 1ULL << (2U -1) = (01)_2 << 1 = (10)_2 = 2  // the -1 in this step means 'After'
    // so repr_ = 2-1 = 1 = CPU
    : repr_((1ULL << (static_cast<uint8_t>(t) - 1)) - 1) {}
  // Public version of DispatchKeySet(uint64_t) API; external users
  // must be explicit when they do this!
  DispatchKeySet(Raw, uint64_t x)
    : repr_(x) {}
  explicit DispatchKeySet(DispatchKey t)
    : repr_(t == DispatchKey::Undefined
              ? 0
              : 1ULL << (static_cast<uint8_t>(t) - 1)) {}
  explicit DispatchKeySet(std::initializer_list<DispatchKey> ks)
    : DispatchKeySet() {
    for (auto k : ks) {
      repr_ |= DispatchKeySet(k).repr_;
    }
  }
  // since one bit represents a specific key, we can use bit op to implement some methods
  // Test if a DispatchKey is in the set
  bool has(DispatchKey t) const {
    TORCH_INTERNAL_ASSERT(t != DispatchKey::Undefined);
    return static_cast<bool>(repr_ & DispatchKeySet(t).repr_);
  }
  // Perform set union
  DispatchKeySet operator|(DispatchKeySet other) const {
    return DispatchKeySet(repr_ | other.repr_);
  }
  // Perform set intersection
  DispatchKeySet operator&(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & other.repr_);
  }
  // Compute the set difference self - other
  DispatchKeySet operator-(DispatchKeySet other) const {
    return DispatchKeySet(repr_ & ~other.repr_);
  }
  // Perform set equality
  bool operator==(DispatchKeySet other) const {
    return repr_ == other.repr_;
  }
  // Add a DispatchKey to the DispatchKey set.  Does NOT mutate,
  // returns the extended DispatchKeySet!
  C10_NODISCARD DispatchKeySet add(DispatchKey t) const {
    return *this | DispatchKeySet(t);
  }
  // Remove a DispatchKey from the DispatchKey set.  This is
  // generally not an operation you should be doing (it's
  // used to implement operator<<)
  C10_NODISCARD DispatchKeySet remove(DispatchKey t) const {
    return DispatchKeySet(repr_ & ~DispatchKeySet(t).repr_);
  }
  // Is the set empty?  (AKA undefined tensor)
  bool empty() const {
    return repr_ == 0;
  }
  uint64_t raw_repr() { return repr_; }
  // Return the type id in this set with the highest priority (i.e.,
  // is the largest in the DispatchKey enum).  Intuitively, this
  // type id is the one that should handle dispatch (assuming there
  // aren't any further exclusions or inclusions).
  DispatchKey highestPriorityTypeId() const {
    // TODO: If I put Undefined as entry 64 and then adjust the
    // singleton constructor to shift from the right, we can get rid of the
    // subtraction here.  It's modestly more complicated to get right so I
    // didn't do it for now.
    return static_cast<DispatchKey>(64 - llvm::countLeadingZeros(repr_));
  }
};
// change to ToString for format compatibility
C10_API std::string toString(DispatchKeySet);
C10_API std::ostream& operator<<(std::ostream&, DispatchKeySet);

// TODO: QUES: keep this?
// Historically, every tensor only had a single DispatchKey, and it was always
// something like CPU, and there wasn't any of this business where TLS
// could cause the DispatchKey of a tensor to change.  But we still have some
// legacy code that is still using DispatchKey for things like instanceof
// checks; if at all possible, refactor the code to stop using DispatchKey in
// those cases.
static inline DispatchKey legacyExtractDispatchKey(DispatchKeySet s) {
  // NB: If you add any extra keys that can be stored in TensorImpl on
  // top of existing "normal" keys like CPU/CUDA, you need to add it
  // here.  At the moment, RequiresGrad (replacement for Variable)
  // is the most likely key that will need this treatment; note that
  // Autograd does NOT need this as it is applied universally
  // (and doesn't show up in TensorImpl)
  return s.highestPriorityTypeId();
}

}