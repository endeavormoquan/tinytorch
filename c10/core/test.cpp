#include <c10/core/MemoryFormat.h>

// typename std::enable_if<true, int>::type t; //正确
// typename std::enable_if<false, int>::type*=nullptr t2; //同上
#include <c10/core/Storage.h>