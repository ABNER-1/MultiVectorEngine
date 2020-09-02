#include <vector>
#include <algorithm>
#include <iostream>

int
main() {
    std::vector<int> new_arrays;
    for (int i = 0; i < 20; ++i) {
        new_arrays.push_back(i);
    }
    std::vector<int> zero_idx{1, 9, 10, 18};
//    std::sort(zero_idx.begin(), zero_idx.end(), std::greater<>());
    for (auto idx : zero_idx) {
        new_arrays.erase(new_arrays.begin() + idx);
//        id_arrays.erase(id_arrays.begin() + idx);
    }
    for (auto i : new_arrays) {
        std::cout << i << std::endl;
    }
}