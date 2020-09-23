#include <chrono>
#include "Utils.h"
#include "MultiVectorCollectionIPNra.h"
#include <omp.h>

namespace milvus {
namespace multivector {

Status
MultiVectorCollectionIPNra::CreateCollection(const std::vector<int64_t>& dimensions,
                                             const std::vector<int64_t>& index_file_sizes) {
    milvus::CollectionParam cp;
    cp.metric_type = metric_type_;
    for (auto i = 0; i < dimensions.size(); ++i) {
        child_collection_names_.emplace_back(GenerateChildCollectionName(i));
        cp.collection_name = child_collection_names_[i];
        cp.dimension = dimensions[i];
        cp.index_file_size = index_file_sizes[i];
        auto status = conn_ptr_->CreateCollection(cp);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::DropCollection() {
    for (auto& child_collection_name: child_collection_names_) {
        auto status = conn_ptr_->DropCollection(child_collection_name);
        if (!status.ok())
            return status;
    }
    return Status::OK();

}

Status
MultiVectorCollectionIPNra::Insert(const std::vector<milvus::multivector::RowEntity>& entity_arrays,
                                   std::vector<int64_t>& id_arrays) {
    std::vector<std::vector<milvus::Entity>> rearranged_arrays
        (child_collection_names_.size(), std::vector<milvus::Entity>(entity_arrays.size(), milvus::Entity()));
    RearrangeEntityArray(entity_arrays, rearranged_arrays, child_collection_names_.size());
    for (auto i = 0; i < rearranged_arrays.size(); ++i) {
        auto status = conn_ptr_->Insert(child_collection_names_[i], "", rearranged_arrays[i], id_arrays);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::Delete(const std::vector<int64_t>& id_arrays) {
    for (auto& child_collection_name : child_collection_names_) {
        auto status = conn_ptr_->DeleteEntityByID(child_collection_name, id_arrays);
        if (!status.ok()) {
            return status;
        }
    }
    conn_ptr_->Flush(child_collection_names_);
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::CreateIndex(milvus::IndexType index_type, const std::string& extra_params) {
    milvus::IndexParam ip;
    ip.index_type = index_type;
    ip.extra_params = extra_params;
    for (auto& child_collection_name: child_collection_names_) {
        ip.collection_name = child_collection_name;
        auto status = conn_ptr_->CreateIndex(ip);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::DropIndex() {
    for (auto& child_collection_name: child_collection_names_) {
        auto status = conn_ptr_->DropIndex(child_collection_name);
        if (!status.ok())
            return status;
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::Flush() {
    return conn_ptr_->Flush(child_collection_names_);
}

Status
MultiVectorCollectionIPNra::SearchImpl(const std::vector<float>& weight,
                                       const std::vector<milvus::Entity>& entity_query,
                                       int64_t topk,
                                       const std::string& extra_params,
                                       QueryResult& query_results,
                                       int64_t tpk) {
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    auto ts = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        std::vector<milvus::Entity> container;
        container.emplace_back(entity_query[i]);
        auto status =
            conn_ptr_->Search(child_collection_names_[i], {}, container, tpk, extra_params, tqrs[i]);
        if (!status.ok())
            return status;
        std::vector<milvus::Entity>().swap(container);
    }
    auto te = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(te - ts).count();

    auto mx_size = tqrs[0][0].ids.size();
    for (auto i = 1; i < child_collection_names_.size(); ++i) {
        mx_size = std::max(tqrs[i][0].ids.size(), mx_size);
    }
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        if (tqrs[i][0].ids.size() == mx_size) continue;
        tqrs[i][0].ids.resize(mx_size, -1);
        tqrs[i][0].distances.resize(mx_size, 0);
    }
    auto ns = std::chrono::high_resolution_clock::now();
    Status stat =
        NoRandomAccessAlgorithmIP(tqrs, query_results, weight, topk) ? Status::OK()
                                                                     : Status(StatusCode::UnknownError,
                                                                              "recall failed!");
    auto es = std::chrono::high_resolution_clock::now();
    auto nra_time = std::chrono::duration_cast<std::chrono::milliseconds>(es - ns).count();
//    std::cerr << search_duration << "; " << nra_time << std::endl;
    return stat;
}

Status
MultiVectorCollectionIPNra::SearchBatch(const std::vector<float>& weight,
                                   const std::vector<RowEntity>& entity_array,
                                   int64_t topk, nlohmann::json& extra_params,
                                   milvus::TopKQueryResult& topk_query_results) {
    std::cout << "not implement yet!" << std::endl;
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::Search(const std::vector<float>& weight,
                                   const std::vector<RowEntity>& entity_array,
                                   int64_t topk, nlohmann::json& extra_params,
                                   milvus::TopKQueryResult& topk_query_results) {
    topk_query_results.resize(entity_array.size());
    topks.clear();
//    #pragma omp parallel for
    for (int q = 0; q < entity_array.size(); ++q) {
        int64_t threshold, tpk;
        tpk = std::max(topk, 512l);
        threshold = 2048;
        bool succ_flag = false;
        do {
            tpk = std::min(threshold, tpk * 2);
            if (extra_params.contains("ef")
                && extra_params["ef"] < tpk) {
                extra_params["ef"] = tpk;
            }
            topk_query_results[q].ids.clear();
            topk_query_results[q].distances.clear();
            auto stat = SearchImpl(weight, entity_array[q], topk, extra_params.dump(), topk_query_results[q], tpk);
            succ_flag = stat.ok();
        } while (!succ_flag && tpk < threshold);
        topks.push_back(tpk);
    }
    return Status::OK();
}

Status
MultiVectorCollectionIPNra::BatchSearch(const std::vector<float>& weight,
                                        const std::vector<RowEntity>& entity_array,
                                        int64_t topk, nlohmann::json& extra_params,
                                        milvus::TopKQueryResult& topk_query_results) {
    auto calc_next_topk = [](int64_t topk) {
        return 2048;
    };
    auto convert_entity = [](std::vector<RowEntity> entity_array) {
        auto nq = entity_array.size();
        auto group_num = entity_array[0].size();
        std::vector<RowEntity> converted_entity_array(group_num, RowEntity(nq));
        for (int i = 0; i < nq; ++i) {
            for (int j = 0; j < group_num; ++j) {
                converted_entity_array[j][i].float_data.swap(entity_array[i][j].float_data);
            }
        }
        return std::move(converted_entity_array);
    };
    topk_query_results.resize(entity_array.size());
    auto converted_entity_array = convert_entity(entity_array);

    while (true) {
        auto tmp_topk = calc_next_topk(topk);
        auto tmp_nq = converted_entity_array[0].size();
        TopKQueryResult tmp_topk_results;
        std::vector<bool> success_flag;
        if (extra_params.contains("ef")
            && extra_params["ef"] < tmp_topk) {
            extra_params["ef"] = tmp_topk;
        }
        BatchSearchImpl(weight, converted_entity_array, topk, tmp_topk,
                        extra_params, tmp_topk_results, success_flag);
        #pragma omp parallel for
        for (int i = 0; i < tmp_nq; ++i) {
            topk_query_results[i].ids.swap(tmp_topk_results[i].ids);
            topk_query_results[i].distances.swap(tmp_topk_results[i].distances);
        }
        break;
    }
    return Status::OK();
}

// converted_entity_array [group num][nq][vector]
Status
MultiVectorCollectionIPNra::BatchSearchImpl(const std::vector<float>& weight,
                                            std::vector<RowEntity>& converted_entity_array,
                                            int64_t topk, int64_t tmp_topk,
                                            nlohmann::json& extra_params, TopKQueryResult& topk_query_results,
                                            std::vector<bool> success_flags) {
    std::vector<TopKQueryResult> tqrs(child_collection_names_.size());
    auto nq = converted_entity_array[0].size();
    success_flags.resize(nq);
    topk_query_results.reserve(nq);
    auto search_start = std::chrono::high_resolution_clock::now();
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        auto status = conn_ptr_->Search(child_collection_names_[i], {},
                                        converted_entity_array[i], tmp_topk, extra_params, tqrs[i]);
        if (!status.ok()) return status;
    }
    auto search_end = std::chrono::high_resolution_clock::now();
    auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(search_end - search_start).count();

    auto nra_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (int j = 0; j < nq; ++j) {
        bool success_flag = NoRandomAccessAlgorithmIP(tqrs, topk_query_results[j], weight, topk);
        success_flags[j] = success_flag;
    }
    auto nra_end = std::chrono::high_resolution_clock::now();
    auto nra_time = std::chrono::duration_cast<std::chrono::milliseconds>(nra_end - nra_start).count();
    std::cerr << search_duration << "; " << nra_time << std::endl;
    return Status::OK();

}

void
MultiVectorCollectionIPNra::GetRowEntityByID(const std::vector<int64_t>& id_arrays,
                                             std::vector<RowEntity>& row_entities) {
    for (auto i = 0; i < child_collection_names_.size(); ++i) {
        auto& collection = child_collection_names_[i];
        std::vector<Entity> entities_data;
        conn_ptr_->GetEntityByID(collection, id_arrays, entities_data);
        for (auto j = 0; j < entities_data.size(); ++j) {
            row_entities[j][i].float_data.swap(entities_data[j].float_data);
        }
    }
}

} // namespace multivector
} // namespace milvus
