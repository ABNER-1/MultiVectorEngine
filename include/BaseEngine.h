#include "MilvusApi.h"


namespace milvus {
namespace multivector {

using Status = milvus::Status;
using RowEntity = std::vector<milvus::Entity>; // multi vector

class BaseEngine {
 public:
    BaseEngine() = default;

    BaseEngine(std::shared_ptr<milvus::Connection> conn_ptr) : conn_ptr_(conn_ptr) {}

    virtual Status
    CreateCollection(const std::string &collection_name,
                     milvus::MetricType metric_type,
                     const std::vector<int64_t> &dimensions,
                     const std::vector<int64_t> &index_file_sizes) = 0;

    virtual Status
    DropCollection(const std::string &collection_name) = 0;

    virtual Status
    Insert(const std::string &collection_name,
           const std::vector<RowEntity> &entity_arrays,
           std::vector<int64_t> &id_arrays) = 0;

    virtual Status
    Delete(const std::string &collection_name, const std::vector<int64_t> &id_arrays) = 0;

    virtual Status
    CreateIndex(const std::string &collection_name, milvus::IndexType index_type, const std::string &param) = 0;

    virtual Status
    DropIndex(const std::string &collection_name) = 0;

    virtual Status
    Search(const std::string &collection_name, const std::vector<float> &weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, milvus::TopKQueryResult &topk_query_results) = 0;

 protected:
    std::shared_ptr<milvus::Connection> conn_ptr_ = nullptr;
};

} // namespace multivector
} //namespace milvus
