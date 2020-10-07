#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/util/Exception.h>

#include <stdexcept>

namespace c10{
/**
 * This legacy enum class defines the set of backends supported by old school,
 * code generated Type-based ATen.  A "backend" in this sense roughly
 * corresponds to the cartesian product of (device type, layout), but restricted
 * only to combinations which we actually have kernels for.  Backend does NOT
 * include dtype.
 *
 * The reason we are sunsetting this enum class is because it doesn't allow for
 * open registration; e.g., if you want to add SparseXLA, you'd have to
 * edit this enum; you wouldn't be able to do it out of tree.  DispatchKey is
 * the replacement for Backend which supports open registration.
 *
 * NB: The concept of 'Backend' here disagrees with the notion of backend
 * exposed to users in torch.backends.  Backend here is something like "CPU"
 * or "SparseCUDA"; backend in torch.backends is something like "MKL" or
 * "CUDNN".
 */
// TODO: NOTE: TLNS(Too Long No See) DispatchKey is the replacement for Backend which supports open registration.
enum class Backend {
  CPU,
  SparseCPU,
  FPGA,
  Undefined,
  NumOptions
};

static inline Backend toSparse(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::SparseCPU;
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend toDense(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::CPU;
    case Backend::SparseCPU:
      return Backend::CPU;
    case Backend::FPGA:
      return Backend::FPGA;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline Backend dispatchKeyToBackend(DispatchKey t) {
  if (t == DispatchKey::CPU) {
    return Backend::CPU;
  } else if (t == DispatchKey::SparseCPU) {
    return Backend::SparseCPU;
  } else if (t == DispatchKey::FPGA) {
    return Backend::FPGA;
  } else if (t == DispatchKey::Undefined) {
    return Backend::Undefined;
  } else {
    AT_ERROR("Unrecognized tensor type ID: ", t);
  }
}

static inline DispatchKey backendToDispatchKey(Backend b) {
  switch (b) {
    case Backend::CPU:
      return DispatchKey::CPU;
    case Backend::SparseCPU:
      return DispatchKey::SparseCPU;
    case Backend::FPGA:
      return DispatchKey::FPGA;
    case Backend::Undefined:
      return DispatchKey::Undefined;
    default:
      throw std::runtime_error("Unknown backend");
  }
}

static inline DeviceType backendToDeviceType(Backend b) {
  switch (b) {
    case Backend::CPU:
      return DeviceType::CPU;
    case Backend::SparseCPU:
      return DeviceType::CPU;
    case Backend::FPGA:
      return DeviceType::FPGA;
    case Backend::Undefined:
      AT_ERROR("Undefined backend is not a valid device type");
    default:
      AT_ERROR("Unknown backend");
  }
}

static inline Backend backendToCPU(Backend b) {
  switch (b) {
    case Backend::CPU:
      return Backend::CPU;
    case Backend::SparseCPU:
      return Backend::SparseCPU;
    // FIGURE: why convert FPGA backend to CPU backend?
    case Backend::FPGA:
      return Backend::CPU;
    case Backend::Undefined:
      return Backend::Undefined;
    default:
      AT_ERROR("Unknown backend");
  }
}

// static inline Backend backendToCUDA(Backend b) {
//   switch (b) {
//     case Backend::CPU:
//     case Backend::SparseCPU:
//     case Backend::Undefined:
//       return Backend::Undefined;
//     default:
//       AT_ERROR("Unknown backend");
//   }
// }

// static inline Backend backendToHIP(Backend b) {
//   switch (b) {
//     case Backend::CPU:
//     case Backend::SparseCPU:
//     case Backend::Undefined:
//       return Backend::Undefined;
//     default:
//       AT_ERROR("Unknown backend");
//   }
// }

static inline const char* toString(Backend b) {
  switch (b) {
    case Backend::CPU:
      return "CPU";
    case Backend::SparseCPU:
      return "SparseCPU";
    case Backend::FPGA:
      return "FPGA";
    default:
      return "UNKNOWN_BACKEND";
  }
}

static inline bool isSparse(Backend b) {
  switch (b) {
    case Backend::SparseCPU:
      return true;
    default:
      return false;
  }
}

}