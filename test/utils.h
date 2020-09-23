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
resetIds();

void
generateIds(int nq, std::vector<int64_t>& id_arrays);

void
generateIds(int nq, std::vector<int64_t>& id_arrays, int base_id);

int
readArrays(const std::string& file_name, const std::vector<int64_t>& dimensions,
           std::vector<milvus::multivector::RowEntity>& row_entities,
           int page_num = 10000, int page = 0);

int
readArraysFromHdf5(const std::string& file_name, const std::vector<int64_t>& dimensions,
                   std::vector<milvus::multivector::RowEntity>& row_entities,
                   int page_num = 10000, int page = 0, const std::string& data_name = "train");

int
readArraysFromSplitedData(const std::vector<std::string>& file_names,
                          const std::vector<int64_t>& dimensions,
                          std::vector<milvus::multivector::RowEntity>& row_entities,
                          int page_num = 10000, int page = 0, int lines=10000);

void
writeBenchmarkResult(const milvus::TopKQueryResult& topk_query_result,
                     const std::string& result_file,
                     float total_time, int topk);

void
showResult(const milvus::TopKQueryResult& topk_query_result);

void
showResultL2(const milvus::TopKQueryResult& topk_query_result);

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              milvus::MetricType metric_type = milvus::MetricType::IP);

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              const nlohmann::json& config,
              milvus::MetricType metric_type = milvus::MetricType::IP);

void
testIndexType(std::shared_ptr<milvus::multivector::MultiVectorEngine> engine,
              milvus::IndexType index_type,
              const nlohmann::json& index_json,
              nlohmann::json& query_json,
              const nlohmann::json& config,
              milvus::MetricType metric_type,
              const std::string &collection_name,
              const std::string &result_file,
              const std::vector<int> &search_args,
              int &file_cnt);

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