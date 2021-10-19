"""
 - Author : Anish Basnet
 - Email : anishbasnetworld@gmail.com
 - Date : April 25, 2020
"""
import os

from tweeter_covid19.probalistic_token_vector_generator import Token2vector
from tweeter_covid19.utils import write_pickle_data, get_file_name, read_pickle_data, mkdir

N_SETS = 10

if __name__ == '__main__':
    backup_path = os.path.join('data', 'distance_vector', 'vectors', 'training_nodes_vectors')
    log_path = os.path.join('data', 'distance_based', 'vectors', 'training_nodes_vectors')
    write_path = os.path.join('data', 'distance_based', 'vectors', 'training_nodes_vectors')
    model_path = os.path.join('data', 'distance_based', 'processing', 'label_based_frequency')

    node_path = os.path.join('data', 'distance_based', 'processing', 'graph_nodes', 'nodes')

    edge_path = os.path.join('data', 'distance_based', 'processing', 'graph_nodes', 'nodes')
    for fold in range(N_SETS):
        freq_joiner_path = os.path.join(model_path, 'set_'+str(fold+1), 'optimizer.pkl')
        freq_model = read_pickle_data(freq_joiner_path)


        write_path_joiner = os.path.join(write_path, 'set_'+str(fold+1))
        mkdir(write_path_joiner)
        write_path_joiner = os.path.join(write_path_joiner, 'direct_training_vectors.pkl')

        backup_path_joiner = os.path.join(backup_path, 'set_'+str(fold+1) + 'backup_1.bak')

        log_path_joiner = os.path.join(log_path, 'set_'+str(fold+1), 'logger.pkl')

        node_joiner_path = os.path.join(node_path, 'set_'+str(fold+1), 'nodes.pkl')
        nodes = read_pickle_data(node_joiner_path)
        log = read_pickle_data(log_path_joiner)

        edge_path_joiner = os.path.join(edge_path, 'set_'+str(fold+1), 'nodes')

        model = Token2vector(freq_model)
        model.fit()
        if log is None:
            log = []
        vectors = dict()
        if os.path.isfile(backup_path_joiner):
            vectors = read_pickle_data(backup_path_joiner)
        for count, node in enumerate(nodes):
            if node in log:
                print("Node : {} - Already executed! Vector Generation "
                      "| Remaining : {}/{} >> Completed Percentage : {}."
                      "".format(node, count, len(nodes), (count / len(nodes)) * 100))

            else:
                index = nodes.index(node)
                vector = model.generate_direct_vectors(node)
                if vector is not None:
                    vectors[node] = vector
                    log.append(node)
                    print("Node : {} - successfully executed! Vector Generation "
                          "| Remaining : {}/{} >> Completed Percentage : {}."
                          "".format(node, count, len(nodes), (count / len(nodes)) * 100))

                    # for backup the pickle file to prevent data loss

                    if count == 0:
                        write_pickle_data(write_path_joiner, vectors)
                        write_pickle_data(log_path_joiner, log)
                    if count % 1000 == 0:
                        if os.path.exists(write_path):
                            # write_dir = get_file_name(write_path_joiner, 1, directory_only=True)
                            # filename = get_file_name(write_path_joiner, 1)
                            # backup_name = 'backup_1.bak'
                            # if os.path.exists(os.path.join(write_path_joiner, backup_name)):
                            #     os.remove(os.path.join(write_path_joiner, backup_name))
                            # os.rename(write_path, os.path.join(write_path_joiner, backup_name))

                            write_pickle_data(write_path_joiner, vectors)
                            write_pickle_data(log_path_joiner, log)
        write_pickle_data(write_path_joiner, vectors)
        write_pickle_data(log_path_joiner, log)
