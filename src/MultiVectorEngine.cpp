#include <memory>
#include <MilvusApi.h>
#include <MultiVectorEngine.h>
#include <MultiVectorCollection.h>
#include <MultiVectorEngine.h>
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
    auto status = createCollectionPtr(collection_name, metric_type);
    if (status.ok()) {
        std::cout << "[ERROR] create collection ptr error" << std::endl;
    }
    return getOrFetchCollectionPtr(collection_name)->CreateCollection(dimensions, index_file_sizes);
}

Status
MultiVectorEngine::DropCollection(const std::string &collection_name) {
    return getOrFetchCollectionPtr(collection_name)->DropCollection();
}

Status
MultiVectorEngine::Insert(const std::string &collection_name,
                          const std::vector<milvus::multivector::RowEntity> &entity_arrays,
                          std::vector<int64_t> &id_arrays) {
    return getOrFetchCollectionPtr(collection_name)->Insert(entity_arrays, id_arrays);
}

Status
MultiVectorEngine::CreateIndex(const std::string &collection_name,
                               milvus::IndexType index_type,
                               const std::string &param) {
    return getOrFetchCollectionPtr(collection_name)->CreateIndex(index_type, param);
}

Status
MultiVectorEngine::DropIndex(const std::string &collection_name) {
    return getOrFetchCollectionPtr(collection_name)->DropIndex();
}

Status
MultiVectorEngine::Search(const std::string &collection_name,
                          const std::vector<float> &weight,
                          const std::vector<RowEntity> &entity_array,
                          int64_t topk, const std::string &extra_params,
                          milvus::TopKQueryResult &topk_query_results) {
    return getOrFetchCollectionPtr(collection_name)->Search(weight,
                                                            entity_array,
                                                            topk,
                                                            extra_params,
                                                            topk_query_results);
}


Status
MultiVectorEngine::createCollectionPtr(const std::string &collection_name,
                                       milvus::MetricType metric_type) {
    MultiVectorCollectionPtr collection_ptr = nullptr;
    if (metric_type == milvus::MetricType::IP) {
        collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
            std::make_shared<MultiVectorCollectionIP>(this->conn_ptr_, collection_name));
    } else if (metric_type == milvus::MetricType::L2) {
        collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
            std::make_shared<MultiVectorCollectionL2>(this->conn_ptr_, collection_name));
//        collection_ptr = std::make_shared<MultiVectorCollectionL2>(this->conn_ptr_, collection_name);
    }
    this->collections_[collection_name] = collection_ptr;
    return Status::OK();
}

MultiVectorCollectionPtr
MultiVectorEngine::getOrFetchCollectionPtr(const std::string &collection_name) {
    auto iter = this->collections_.find(collection_name);
    if (iter != this->collections_.end()) {
        return this->collections_[collection_name];
    }
    // fetch information and create from milvus or from storage.
    return nullptr;
}

} // namespace multivector
} // namespace milvus
