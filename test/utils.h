#pragma once
#include "MultiVectorEngine.h"
#include <random>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <numeric>

void
normalizeVector(milvus::Entity& entity);

void
generateVector(int64_t dim, milvus::Entity& entity);

void
generateArrays(int nq, const std::vector<int64_t>& dimensions,
               std::vector<milvus::multivector::RowEntity>& row_entities);

void
generateIds(int nq, std::vector<int64_t>& id_arrays);

void
showResult(const milvus::TopKQueryResult& topk_query_result);

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              const nlohmann::json& query_json,
              milvus::MetricType metric_type = milvus::MetricType::IP);

