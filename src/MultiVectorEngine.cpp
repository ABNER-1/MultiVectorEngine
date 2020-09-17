#include <memory>
#include <MilvusApi.h>
#include <MultiVectorEngine.h>
#include <MultiVectorCollection.h>
#include <MultiVectorEngine.h>
#include <nlohmann/json.hpp>
#include "MultiVectorCollectionIP.h"
#include "MultiVectorCollectionL2.h"
#include "MultiVectorCollectionIPNra.h"

namespace milvus {
namespace multivector {

MultiVectorEngine::MultiVectorEngine(const std::string& ip, const std::string& port) {
    this->conn_ptr_ = milvus::Connection::Create();
    ConnectParam param{ip, port};
    this->conn_ptr_->Connect(param);
}

Status
MultiVectorEngine::CreateCollection(const std::string& collection_name,
                                    milvus::MetricType metric_type,
                                    const std::vector<int64_t>& dimensions,
                                    const std::vector<int64_t>& index_file_sizes,
                                    const std::string& strategy) {
    auto status = createCollectionPtr(collection_name, metric_type, strategy);
    if (!status.ok()) {
        std::cout << "[ERROR] create collection ptr error: " << status.message() << std::endl;
    }
    return getOrFetchCollectionPtr(collection_name)->CreateCollection(dimensions, index_file_sizes);
}

Status
MultiVectorEngine::DropCollection(const std::string& collection_name) {
    return getOrFetchCollectionPtr(collection_name)->DropCollection();
}

Status
MultiVectorEngine::Insert(const std::string& collection_name,
                          const std::vector<milvus::multivector::RowEntity>& entity_arrays,
                          std::vector<int64_t>& id_arrays) {
    return getOrFetchCollectionPtr(collection_name)->Insert(entity_arrays, id_arrays);
}

Status
MultiVectorEngine::Delete(const std::string& collection_name,
                          const std::vector<int64_t>& id_arrays) {
    return getOrFetchCollectionPtr(collection_name)->Delete(id_arrays);
}

Status
MultiVectorEngine::CreateIndex(const std::string& collection_name,
                               milvus::IndexType index_type,
                               const std::string& param) {
    return getOrFetchCollectionPtr(collection_name)->CreateIndex(index_type, param);
}

Status
MultiVectorEngine::DropIndex(const std::string& collection_name) {
    return getOrFetchCollectionPtr(collection_name)->DropIndex();
}

Status
MultiVectorEngine::Flush(const std::string& collection_name) {
    return getOrFetchCollectionPtr(collection_name)->Flush();
}

Status
MultiVectorEngine::Search(const std::string& collection_name,
                          const std::vector<float>& weight,
                          const std::vector<RowEntity>& entity_array,
                          int64_t topk, nlohmann::json& extra_params,
                          milvus::TopKQueryResult& topk_query_results) {
    return getOrFetchCollectionPtr(collection_name)->Search(weight, entity_array, topk,
                                                            extra_params, topk_query_results);
}

Status
MultiVectorEngine::createCollectionPtr(const std::string& collection_name,
                                       milvus::MetricType metric_type,
                                       const std::string& strategy) {
    MultiVectorCollectionPtr collection_ptr = nullptr;
    if (metric_type == milvus::MetricType::IP) {
        if (strategy != "default") {
            collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
                std::make_shared<MultiVectorCollectionIPNra>(this->conn_ptr_, collection_name));
        } else {
            collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
                std::make_shared<MultiVectorCollectionIP>(this->conn_ptr_, collection_name));
        }

    } else if (metric_type == milvus::MetricType::L2) {
        collection_ptr = std::static_pointer_cast<MultiVectorCollection>(
            std::make_shared<MultiVectorCollectionL2>(this->conn_ptr_, collection_name));
//        collection_ptr = std::make_shared<MultiVectorCollectionL2>(this->conn_ptr_, collection_name);
    }
    this->collections_[collection_name] = collection_ptr;
    return Status::OK();
}

MultiVectorCollectionPtr
MultiVectorEngine::getOrFetchCollectionPtr(const std::string& collection_name) {
    auto iter = this->collections_.find(collection_name);
    if (iter != this->collections_.end()) {
        return this->collections_[collection_name];
    }
    // todo: fetch information and create from milvus or from storage.
    std::cout << "[ERROR] NO collection exist: " << collection_name << std::endl;
    return nullptr;
}

void
MultiVectorEngine::CalcDistance(const std::string& collection_name,
                                const std::vector<int64_t>& id_arrays,
                                const RowEntity& query_entities,
                                const std::vector<float>& weights,
                                QueryResult& query_result) {
    std::vector<RowEntity> row_entities;
    GetRowEntityByID(collection_name, id_arrays, row_entities);
    CalcDistanceImpl(row_entities, id_arrays, query_entities, weights, query_result);
}

void
MultiVectorEngine::GetRowEntityByID(const std::string& collection_name,
                                    const std::vector<int64_t>& id_arrays,
                                    std::vector<RowEntity>& row_entities) {
    row_entities.resize(id_arrays.size());
    getOrFetchCollectionPtr(collection_name)->GetRowEntityByID(id_arrays, row_entities);
}

void
MultiVectorEngine::CalcDistanceImpl(const std::vector<RowEntity>& row_entities,
                                    const std::vector<int64_t>& id_arrays,
                                    const RowEntity& query_entities,
                                    const std::vector<float>& weights,
                                    milvus::QueryResult& query_result) {
    query_result.ids = id_arrays;
    int group_num = row_entities.size();
    int topk = row_entities[0].size();
    query_result.ids.resize(topk);
    for (auto j = 0; j < topk; ++j) {
        for (auto i = 0; i < group_num; ++i) {
            auto& data = row_entities[i][j].float_data;
            for (auto k = 0; k < data.size(); ++k) {
                auto tmp = data[k] - query_entities[i].float_data[k];
                query_result.distances[j] += weights[i] * (tmp * tmp);
            }
        }
    }
}

} // namespace multivector
} // namespace milvus
