import argparse
from runner.runner import Runner 
from config.config import init_cfg
from utils.misc import str_list
from utils import dist_utils
import torch

def main():

    # Create an argument parser for command line interaction
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")

    # Arguments for setting up the environment and training configuration
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync batch normalization')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file", type=str)
    parser.add_argument("--task", default="train", help="tasks like train, val, val_gen", type=str)
    parser.add_argument("--prefix", default="exp", type=str)
    parser.add_argument("--no_cuda", action='store_true', help='Disable CUDA')
    parser.add_argument("--no_eval", action='store_true', help='Disable evaluation')
    parser.add_argument("--short_val", action='store_true', help='Short validation run')
    parser.add_argument("--gen_num", default=400, type=int, help='Number of samples to generate')
    parser.add_argument("--part_id", default=2, type=int, help='Part ID for processing')
    parser.add_argument('--interpolation_dir', default="./", type=str, help='Directory for interpolation data')
    parser.add_argument("--param_sample_num", default=10, type=int, help='Number of parameter samples')
    parser.add_argument("--save_dir", default=".", type=str, help='Directory to save outputs')

    # Parse the arguments
    args = parser.parse_args()

    # Set the device based on CUDA availability
    device = 'cpu' if args.no_cuda else 'cuda'

    # Assert to ensure the task is one of the supported types
    assert args.task in ["train", "val", "val_gen", 'interpolation'], f"{args.task} not supported, please choose from [train, val, test, vis_test]"
    
    # Initialize the configuration from a file if provided
    if args.config_file:
        init_cfg(args.config_file)

    # Set CUDNN backend options if CUDA is enabled
    if device == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Initialize distributed environment if required
    if args.launcher == 'none':
        args.distributed = False
        args.world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    # Initialize and run the runner based on the specified task
    runner = Runner(device, args)
    if args.task == "train":
        runner.run()
    elif args.task == "interpolation":
        import pickle 
        data = pickle.load(open(args.interpolation_dir, "rb"))
        runner.interpolate_two_sets(data['set1'], data['set2'], args.part_id)
    elif args.task == "val":
        runner.val()
    elif args.task == "val_gen":
        runner.generate_samples(args.gen_num, args.param_sample_num)

if __name__ == "__main__":
    main()
