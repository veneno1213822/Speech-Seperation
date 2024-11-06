import argparse
import importlib
# import torch

def main_train():

    # Parse args
    parser = argparse.ArgumentParser(
        description="Command to start PIT training, configured by .yaml files")
    parser.add_argument(
        "--model",
        type=str,
        dest="model",
        default="SepReformer_Large_DM_WSJ0",
        help="Insert model name")
    parser.add_argument(
        "--engine-mode",
        choices=["train", "test", "test_save", "infer_sample"],
        default="train",
        help="This option is used to chooose the mode")
    parser.add_argument(
        "--sample-file",
        type=str,
        default=None,
        help="directoy for sample audio")
    parser.add_argument(
        "--out-wav-dir",
        type=str,
        default=None,
        help="This option is used to specficy save directory for output wav file in test_wav mode")
    args = parser.parse_args()

    # Call target model
    main_module = importlib.import_module(f"models.{args.model}.main")
    main_module.main(args)

if __name__ == "__main__":
    main_train()
    # configs.yaml里gpuid设置使用哪个GPU