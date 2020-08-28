#include "MilvusApi.h"
#include <string>

namespace milvus {
namespace multivector {

void GenCollectionSchema(const std::string& collection_name, milvus::HMapping& mapping);

} // namespace multivector
} // namespace milvus
