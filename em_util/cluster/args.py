import argparse

def get_parser():
    """
    The function `get_arguments()` is used to parse command line arguments
    :return: The function `get_arguments` returns the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="argument parser"
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="task",
        default="",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        type=str,
        help="cluster name",
        default="harvard",
    )
    parser.add_argument(
        "-s",
        "--cmd",
        type=str,
        help="slurm command",
        default="",
    ) 
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        help="conda environment name",
        default="imu",
    )
    parser.add_argument(
        "-ji",
        "--job-id",
        type=int,        
        default=0,
    )    
    parser.add_argument(
        "-jn",
        "--job-num",
        type=int,        
        default=1,
    )
    parser.add_argument(
        "-cn",
        "--chunk-num",
        type=int,        
        default=1,
    )
    parser.add_argument(
        "-n",
        "--neuron",
        type=str,
        help="neuron ids",
        default="",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        type=str,
        help="downsample ratio",
        default="1,1,1",
    )
    parser.add_argument(
        "-cp",
        "--partition",
        type=str,
        help="",
        default="lichtman",
    )
    parser.add_argument(
        "-cm",
        "--memory",
        type=str,
        help="",
        default="50GB",
    )
    parser.add_argument(
        "-ct",
        "--run-time",
        type=str,
        help="",
        default="0-12:00",
    )
    parser.add_argument(
        "-cg",
        "--num_gpu",
        type=int,
        help="",
        default=-1,
    )
    return parser

