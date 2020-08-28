#include <unordered_map>
#include "MilvusApi.h"
#include "Status.h"
#include "BaseEngine.h"


namespace milvus {
namespace multivector {

class MultiVectorEngine : BaseEngine {
 public:
    MultiVectorEngine() {}
    MultiVectorEngine(const std::string &ip, const std::string &port) {}

    Status
    CreateCollection(std::string collection_name, milvus::MetricType metric_type,
                     std::vector<int64_t> dimensions,
                     std::vector<int64_t> index_file_sizes) override;

    Status
    DropCollection(std::string collection_name) override {
        // todo: 1 find collection_name in collections_
        // todo: 2 for id in collections_[collection_name].second:
        //  do milvus.dropindex(GenerateChildCollectionName(collection_name, id));
    }

    Status
    Insert(const std::string &collection_name,
           const std::vector<RowEntity> &entity_arrays,
           std::vector<int64_t> &id_arrays) override;

    Status
    Delete(const std::string &collection_name, std::vector<int64_t> &id_arrays) override;

    Status
    CreateIndex(const std::string &collection_name, milvus::MetricType index_type, std::string param) override;

    Status
    DropIndex(const std::string &collection_name) override;

    Status
    Search(const std::string &collection_name, std::vector<float> weight,
           const std::vector<std::vector<milvus::Entity>> &entity_array,
           int64_t topk, milvus::TopKQueryResult &topk_query_results) override;

 private:
    static std::string
    GenerateChildCollectionName(const std::string &collection_prefix, int64_t idx) {
        return collection_prefix + "_" + std::to_string(idx);
    }

 private:
    // todo: use map first, edit it later -- serialization to file or db later
    std::unordered_map<std::string, milvus::MetricType> metric_map_;
    // maintain collection list for all collections_
    // collections_.first is the user provided collection name
    // collections_.second is the child collection(s) list
    std::unordered_map<std::string, std::vector<int64_t>> collections_;
};

} // namespace multivector
} // namespace milvus
