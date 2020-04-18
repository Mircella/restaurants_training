import os
import pickle


def write_df_to_csv (data_dir, file_name, data_frame):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_frame.to_csv(os.path.join(data_dir, file_name), header=False, index=False)


def write_to_txt (data_dir, file_name, object_to_write):
    text = str(object_to_write)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    f = open(file_name, 'a')
    f.write(text)
    f.close()


def read_txt_file(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    result = ""
    try:
        with open(path, "r") as file:
            result = file.read()
        print("Read text data from file:", file_name)
    except:
        print("Failed to read text data from file:", file_name)
    return result


def read_from_pickle_binary(data_dir, file_name):
    cache_data = None
    if file_name is not None:
        try:
            with open(os.path.join(data_dir, file_name), "rb") as f:
                cache_data = pickle.load(f)
            print("Read data from pickle binary file:", file_name)
        except:
            print("Failed to read data from pickle binary file:", file_name)
    return cache_data


def write_to_pickle_binary(data_dir, file_name, data):
    if file_name is not None:
        try:
            with open(os.path.join(data_dir, file_name), "wb") as f:
                pickle.dump(data, f)
            print("Wrote binary data to pickle file:", file_name)
        except:
            print("Failed to write data to pickle binary file:", file_name)