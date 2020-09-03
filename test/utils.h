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

int
readArrays(const std::string& file_name, const std::vector<int64_t>& dimensions,
           std::vector<milvus::multivector::RowEntity>& row_entities,
           int page_num = 10000, int page = 0);

int
readArraysFromHdf5(const std::string& file_name, const std::vector<int64_t>& dimensions,
                   std::vector<milvus::multivector::RowEntity>& row_entities,
                   int page_num = 10000, int page = 0, const std::string& data_name = "train");

void
showResult(const milvus::TopKQueryResult& topk_query_result);

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              const nlohmann::json& query_json,
              milvus::MetricType metric_type = milvus::MetricType::IP);

void
testIndexTypeIP(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
                milvus::IndexType index_type,
                const nlohmann::json& index_json,
                const nlohmann::json& query_json,
                milvus::MetricType metric_type = milvus::MetricType::IP);

void
loadDataFromHdf5(const std::string& filename,
                 std::vector<std::vector<float>>& vector_data,
                 unsigned& num, unsigned& dim, int page_num = 10000, int page = 0,
                 const std::string& data_name = "train");

void
loadDataFromFvec(const std::string& filename,
                 std::vector<std::vector<float>>& vector_data,
                 unsigned& num, unsigned& dim,
                 int page_num = 10000, int page = 0);

void
split_data(const std::vector<std::vector<float>>& raw_data,
           std::vector<std::vector<milvus::Entity>>& splited_data,
           const std::vector<int64_t>& dims);

void
compareResultWithH5(const milvus::TopKQueryResult& topk_query_result,
                    const std::string& h5_file_name);