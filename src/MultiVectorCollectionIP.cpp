#include "MultiVectorCollectionIP.h"


namespace milvus {
namespace multivector {

Status
MultiVectorCollectionIP::CreateCollection(const std::vector<int64_t> &dimensions,
                                          const std::vector<int64_t> &index_file_sizes) {

    int64_t new_dims = 0;
    for (auto dims : dimensions) {
        new_dims += dims;
    }
    if (new_dims > (1 << 15)) {
        // todo: should not use std::cout, replace it later.
        std::cout << "[ERROR] total dims has excced the dim limit, new dims is " << new_dims << std::endl;
    }

    milvus::CollectionParam param
        {this->collection_name_, new_dims, index_file_sizes[0],
         this->metric_type_};
    auto status = this->conn_ptr_->CreateCollection(param);
    if (!status.ok()) {
        return status;
    }
    return Status::OK();
}


Status
MultiVectorCollectionIP::DropCollection() {
    return this->conn_ptr_->DropCollection(this->collection_name_);
}

Status
MultiVectorCollectionIP::Insert(const std::vector<milvus::multivector::RowEntity> &entity_arrays,
                                std::vector<int64_t> &id_arrays) {
    auto nq = entity_arrays.size();
    // validate id array size is nq
    if (id_arrays.size() != nq) {
        std::string err_msg = "id arrays do not have the same size with entity_arrays";
        return Status(StatusCode::InvalidAgument, err_msg);
    }
    std::vector<milvus::Entity> new_arrays(nq);
    auto status = this->mergeRowEntityFromEntites(entity_arrays, new_arrays);
    if (status.ok()) {
        //
        std::cout << "maybe never wrong." << std::endl;
    }
    return this->conn_ptr_->Insert(this->collection_name_, "", new_arrays, id_arrays);
}

Status
MultiVectorCollectionIP::Delete(const std::vector<int64_t> &id_arrays) {
    return this->conn_ptr_->DeleteEntityByID(this->collection_name_, id_arrays);
}

Status
MultiVectorCollectionIP::CreateIndex(milvus::IndexType index_type, const std::string &extra_params) {
    IndexParam index_param{this->collection_name_, index_type, extra_params};
    return this->conn_ptr_->CreateIndex(index_param);
}


Status
MultiVectorCollectionIP::DropIndex() {
    return this->conn_ptr_->DropIndex(this->collection_name_);
}

Status
MultiVectorCollectionIP::Search(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, milvus::TopKQueryResult &topk_query_results) {
    std::vector<milvus::Entity> new_arrays(entity_array.size());
    auto status = this->mergeRowEntityFromEntites(entity_array, new_arrays);
    if (status.ok()) {
        std::cout << "maybe never wrong." << std::endl;
    }
    this->conn_ptr_->Search(this->collection_name_, {}, new_arrays, topk, "", topk_query_results);

    return Status::OK();
}


Status
MultiVectorCollectionIP::mergeRowEntityFromEntites(const std::vector<RowEntity> &entity_arrays,
                                                   std::vector<milvus::Entity> &new_entities) {

    for (auto i = 0; i < entity_arrays.size(); ++i) {
        auto &tmp_row_entity = entity_arrays[i];
        auto &target_row_entity = new_entities[i];
        for (auto &entities : tmp_row_entity) {
            auto &target_float_vector = target_row_entity.float_data;
            auto &tmp_float_vector = entities.float_data;
            target_float_vector.insert(target_float_vector.end(), tmp_float_vector.begin(), tmp_float_vector.end());

//            ignore binary data
//            auto &target_binary_vector = target_row_entity.binary_data;
//            auto &tmp_binary_vector = entities.binary_data;
//            target_binary_vector.insert(target_binary_vector.end(), tmp_binary_vector.begin(), tmp_binary_vector.end());
        }
    }
    return Status::OK();
}


} // namespace multivector
} // namespace milvus