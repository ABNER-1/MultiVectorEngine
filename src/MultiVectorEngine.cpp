#include <memory>
#include <MilvusApi.h>
#include <MultiVectorEngine.h>
#include <MultiVectorCollection.h>
#include "MultiVectorEngine.h"
#include "MultiVectorCollectionIP.h"
#include "MultiVectorCollectionL2.h"


namespace milvus {
namespace multivector {

MultiVectorEngine::MultiVectorEngine(const std::string &ip, const std::string &port) {
    this->conn_ptr_ = milvus::Connection::Create();
    ConnectParam param{ip, port};
    this->conn_ptr_->Connect(param);
}

Status
MultiVectorEngine::CreateCollection(const std::string &collection_name,
                                    milvus::MetricType metric_type,
                                    const std::vector<int64_t> &dimensions,
                                    const std::vector<int64_t> &index_file_sizes) {
    MultiVectorCollectionPtr collection_ptr = nullptr;
    if (metric_type == milvus::MetricType::IP) {
        collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
            std::make_shared<milvus::multivector::MultiVectorCollectionIP>(this->conn_ptr_, collection_name));
    } else if (metric_type == milvus::MetricType::L2) {
        collection_ptr = std::make_shared<milvus::multivector::MultiVectorCollectionL2>(this->conn_ptr_, collection_name);
    }
    this->collections_[collection_name] = collection_ptr;
    return collection_ptr->CreateCollection(dimensions, index_file_sizes);

}


} // namespace multivector
} // namespace milvus
