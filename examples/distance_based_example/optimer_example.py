"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 24, 2020
"""
import os

from tweeter_covid19.searching_optimizer import Optimizer, Frequency_generator
from tweeter_covid19.utils import get_all_directory, read_data_from_file, write_pickle_data, mkdir

N_SETS = 10

if __name__ == '__main__':
    data_path = os.path.join('data', 'distance_based', 'processing', 'label_based_corpus')
    write_path = os.path.join('data', 'distance_based', 'processing', 'label_based_frequency')

    for fold in range(N_SETS):
        reader_joiner_path = os.path.join(data_path, 'set_' + str(fold + 1))
        writer_joiner_path = os.path.join(write_path, 'set_' + str(fold + 1))

        mkdir(writer_joiner_path)

        labels = get_all_directory(reader_joiner_path)
        model = Optimizer(labels)

        for label in labels:
            tokens = read_data_from_file(os.path.join(reader_joiner_path, label, label + '.txt'))
            model.process(label, tokens)

        freq_model = Frequency_generator(model)
        freq_model.fit()
        freq_model.generate_frequency(verbose=True)
        write_pickle_data(os.path.join(writer_joiner_path, 'optimizer.pkl'), freq_model)
