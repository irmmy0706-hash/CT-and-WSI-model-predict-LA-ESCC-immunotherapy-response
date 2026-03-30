# ?????????????????????:
import os
import pickle
from matplotlib import pyplot as plt




# ?? test_metrics ? logs ??
def save_test_metrics_to_log(test_metrics, log_dir='logs'):
    # ?? logs ????
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ??? .pkl ??
    log_file_path = os.path.join(log_dir, 'test_metrics.pkl')
    with open(log_file_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    print(f"Test metrics saved to {log_file_path}")
def save_best_metrics_to_log(test_metrics, log_dir='logs'):
    # ?? logs ????
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ??? .pkl ??
    log_file_path = os.path.join(log_dir, 'best_test_metrics.pkl')
    with open(log_file_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    print(f"Test metrics saved to {log_file_path}")

# ?????? test_metrics
def load_test_metrics_from_log(log_dir='logs'):
    log_file_path = os.path.join(log_dir, 'test_metrics.pkl')
    if os.path.exists(log_file_path):
        print(f"metrics found in {log_file_path}")
        with open(log_file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"No saved metrics found in {log_file_path},start testing")
        return None


def save_data_to_cache(cache_path, data):
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)


def load_data_from_cache(cache_path):
    with open(cache_path, 'rb') as f:
        return pickle.load(f)