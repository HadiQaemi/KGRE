from KGRE.utils import load_config_as_namespace, prepare_load_data
from KGRE.classify import Classify 
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Multimodal Classifier")
    parser.add_argument('--log_dir', type=str, default='logs', help='Path to the log directory')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the log directory')
    return parser
    
parser = create_parser()
args = parser.parse_args()
config = load_config_as_namespace(args.config)
config.log_dir = args.log_dir

if __name__ == "__main__":
    model = Classify(config)
    model()