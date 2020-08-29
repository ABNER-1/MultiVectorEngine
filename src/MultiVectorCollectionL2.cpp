#include "MultiVectorCollectionL2.h"


namespace milvus {
namespace multivector {

Status
MultiVectorCollectionL2::CreateCollection(const std::vector<int64_t> &dimensions,
                                          const std::vector<int64_t> &index_file_sizes) {
    return Status::OK();

}

Status
MultiVectorCollectionL2::DropCollection() {
    return Status::OK();
}

Status
MultiVectorCollectionL2::Insert(const std::vector<milvus::multivector::RowEntity> &entity_arrays,
                                std::vector<int64_t> &id_arrays) {
    return Status::OK();
}

Status
MultiVectorCollectionL2::Delete(const std::vector<int64_t> &id_arrays) {
    return Status::OK();
}

Status
MultiVectorCollectionL2::CreateIndex(milvus::IndexType index_type, const std::string &extra_params) {
    return Status::OK();
}


Status
MultiVectorCollectionL2::DropIndex() {
    return Status::OK();
}

Status
MultiVectorCollectionL2::Search(const std::vector<float> &weight,
                                const std::vector<RowEntity> &entity_array,
                                int64_t topk, const std::string &extra_params,
                                milvus::TopKQueryResult &topk_query_results) {
    return Status::OK();
}

} // namespace multivector
} // namespace milvus
