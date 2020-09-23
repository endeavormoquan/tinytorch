#include <c10/core/Device.h>
#include <c10/util/Exception.h>

#include <string>

namespace c10{
std::string Device::str() const{
    std::string str = DeviceTypeName(type(), true);
    if (has_index()){
        // TODO: LEARN: push back a char, append a string
        str.push_back(':');
        str.append(to_string(index()));
    }
    return str;
}

std::ostream& operator<<(std::ostream& stream, const Device& device){
    stream<<device.str();
    return stream;
}
}