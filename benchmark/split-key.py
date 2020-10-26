import pickle


def write2pickle(data, file_name):
    with open(file_name, "wb") as out:
        pickle.dump(data, out)


def split_func(part_number):
    with open(key_file_name, 'rb') as file:
        data = pickle.load(file)
        raw_length = len(data)
        print(raw_length)
        step = raw_length // part_number
        for i in range(part_number):
            tmp_data = data[i * step: (i + 1) * step]
            if i == part_number - 1:
                tmp_data = data[i * step:]
            tmp_file = key_file_prefix + str(i) + ".pkl"
            write2pickle(tmp_data, tmp_file)


if __name__ == "__main__":
    key_file_prefix = "./test_keys"
    key_file_name = key_file_prefix + ".pkl"
    part_num = 10
    split_func(part_num)
