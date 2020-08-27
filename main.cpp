#include <iostream>
#include <unordered_map>
#include "MilvusApi.h"
#include "Status.h"

namespace mv {

struct MultiEntitiesCollection {
    std::string collection_name;
    milvus::MetricType metric_type;
    std::vector<int64_t> dimensions;
    std::vector<int64_t> index_file_sizes;
};

class MultiVectorEngine {
 public:
    MultiVectorEngine(const std::string& ip, const std::string& port) {
        this->conn_ptr_ = milvus::Connection::Create();
        milvus::ConnectParam param = {ip, port};
        this->conn_ptr_->Connect(param);
    }

    ~MultiVectorEngine() {
        milvus::Connection::Destroy(this->conn_ptr_);
    }

    milvus::Status
    CreateCollection(MultiEntitiesCollection collection_info);

    milvus::Status
    DropCollection(std::string collection_name);

    milvus::Status
    Insert(const std::string& collection_name,
           const std::string& partition_tag,
           const std::vector<std::vector<milvus::Entity>>& entity_arrays,
           std::vector<int64_t>& id_arrays);

    milvus::Status
    Delete(std::string collection_name, std::vector<int64_t>& id_arrays);

    milvus::Status
    CreateIndex(std::string collection_name, int64_t index_type, std::string param);

    milvus::Status
    DropIndex(std::string collection_name);

    milvus::Status
    Search(const std::string& collection_name, std::vector<float> weight,
           const std::vector<std::vector<milvus::Entity>>& entity_array,
           int64_t topk, milvus::TopKQueryResult& topk_query_results);

 private:
    static std::string
    GenerateChildCollectionName(const std::string& collection_prefix, int64_t idx) {
        return collection_prefix + "_" + std::to_string(idx);
    }

 private:
    std::shared_ptr<milvus::Connection> conn_ptr_ = nullptr;
    // use map first, edit it later
    std::unordered_map<std::string, milvus::MetricType> metric_map;
};

milvus::Status
MultiVectorEngine::CreateCollection(MultiEntitiesCollection collection_info) {
    this->metric_map[collection_info.collection_name] = collection_info.metric_type;
    if (collection_info.metric_type == milvus::MetricType::L2) {
        for (auto i = 0; i < collection_info.dimensions.size(); ++i) {
            std::string tmp_collection = GenerateChildCollectionName(collection_info.collection_name, i);
            milvus::CollectionParam param
                {tmp_collection, collection_info.dimensions[i], collection_info.index_file_sizes[i],
                 collection_info.metric_type};
            auto status = this->conn_ptr_->CreateCollection(param);
            if (!status.ok()) {
                return status;
            }
        }
    } else if (collection_info.metric_type == milvus::MetricType::IP) {
        int64_t new_dims = 0;
        for (auto dims : collection_info.dimensions) {
            new_dims += dims;
        }
        milvus::CollectionParam param
            {collection_info.collection_name, new_dims, collection_info.index_file_sizes[0],
             collection_info.metric_type};
        auto status = this->conn_ptr_->CreateCollection(param);
        if (!status.ok()) {
            return status;
        }
    } else {
        std::cout << "unsupported metric" << std::endl;
    }

    return milvus::Status::OK();
}

milvus::Status
MultiVectorEngine::Insert(const std::string& collection_name,
                          const std::string& partition_tag,
                          const std::vector<std::vector<milvus::Entity> >& entity_arrays,
                          std::vector<int64_t>& id_array) {
    for (auto i = 0; i < entity_arrays.size(); ++i) {
        auto& entity_array = entity_arrays[i];
        std::string tmp_collection_name = GenerateChildCollectionName(collection_name, i);
        auto status = this->conn_ptr_->Insert(tmp_collection_name, "", entity_array, id_array);
        if (!status.ok()) {
            return status;
        }
    }
    return milvus::Status::OK();
}

milvus::Status
MultiVectorEngine::Search(const std::string& collection_name, std::vector<float> weight,
                          const std::vector<std::vector<milvus::Entity>>& entity_array,
                          int64_t topk, milvus::TopKQueryResult& topk_query_results) {
    return milvus::Status();
}

} // namespace mv


int
main() {

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
