#include <c10/core/DispatchKey.h>

namespace c10 {

const char* toString(DispatchKey t) {
  switch (t) {
    case DispatchKey::Undefined:
      return "Undefined";
    case DispatchKey::CPU:
      return "CPU";
    case DispatchKey::SparseCPU:
      return "SparseCPU";
    case DispatchKey::Autograd:
      return "Autograd";
    case DispatchKey::BackendSelect:
      return "BackendSelect";
    case DispatchKey::Batched:
      return "Batched";
    case DispatchKey::TESTING_ONLY_GenericMode:
      return "TESTING_ONLY_GenericMode";
    case DispatchKey::Autocast:
      return "Autocast";
    case DispatchKey::TESTING_ONLY_GenericWrapper:
      return "TESTING_ONLY_GenericWrapper";
    case DispatchKey::Profiler:
      return "Profiler";
    case DispatchKey::Named:
      return "Named";
    case DispatchKey::Tracer:
      return "Tracer";
    case DispatchKey::Meta:
      return "Meta";
    default:
      return "UNKNOWN_TENSOR_TYPE_ID";
  }
}

std::ostream& operator<<(std::ostream& str, DispatchKey rhs) {
  return str << toString(rhs);
}

} // namespace c10