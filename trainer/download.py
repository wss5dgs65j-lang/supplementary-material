# download.py

# ================================================================================== #
#   This script is to prepare the data.                                              #
#   We run this script once locally or in the cloud, save the output,                #
#   and then all the training jobs will read from that saved data.                   #
#                                                                                    #
#   Run (currently in fllm-project/code directory)                                   #
#       python download.py --data-dir ../data --dataset-name tinystories simplestories             #
# ================================================================================== #

import argparse
import os

from datasets import DatasetDict
from trainer.YANData import download, tokenize_or_load, merge_datasets

from rich.console import Console
from rich.panel import Panel


def main():
    """Download, tokenize, and save the tokenized data"""
    parser = argparse.ArgumentParser(description=f"Download and tokenize data.")
    parser.add_argument(
        '--data-dir', type=str, required=True
    )
    parser.add_argument(
        '--dataset-name', type=str, nargs="+", required=True
    )
    parser.add_argument(
        '--model-name', type=str, required=False, default='meta-llama/Llama-3.1-8B'
    )
    parser.add_argument(
        '--max-length', type=int, required=False, default=2048
    )
    parser.add_argument(
        '--tokenize-batch-size', type=int, required=False, default=1024
    )
    
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    max_length = args.max_length
    tokenize_batch_size = args.tokenize_batch_size

    text_key = "text"

    console = Console()

    # ------ 1. Create output directory ------
    data_dir = os.path.join(args.data_dir, "-".join(args.dataset_name))
    os.makedirs(data_dir, exist_ok=True)
    console.print(f"[green]✓ Data will be saved to directory:[/green] [bold cyan]{data_dir}[/bold cyan]")

    # ------ 2. Download data ------
    datasets = []
    for ds_name in dataset_name:
        with console.status(f"[bold green]Downloading {ds_name} dataset...", spinner="arc") as status:
            ds = download(ds_name, val_ratio=0.05, test_ratio=0.05, seed=42)
            datasets.append(ds)
        console.print(f"[green]✓ Downloaded {ds_name} dataset successfully![/green]")
        console.print(f"[bold blue]{ds_name} structure:[/bold blue]\n{ds}")
    
    # ------ 3. Merge data ------
    with console.status(f"[bold green]Merging datasets {dataset_name} ...", spinner="arc") as status:
        dataset_merged = merge_datasets(datasets, seed=42)
    console.print(f"[green]✓ Merged datasets {dataset_name} successfully![/green]")


    # ------ 4. Tokenize the data ------ 
    with console.status(f"[bold green]Using {os.cpu_count()} cores. Tokenizing and writing data...", spinner="arc") as status:
        assert isinstance(dataset_merged, DatasetDict)
        tokenize_or_load(dataset_merged, model_name, text_key, data_dir, max_length, 
                         tokenize_batch_size, False, console)

    console.print(Panel.fit(
        "[bold green]All data has been successfully downloaded and tokenized![/bold green]" \
        "\nReady for model training.",
        title="[bold yellow]Preprocessing Complete[/bold yellow]",
        border_style="blue"
    ))


if __name__ == '__main__':
    main()
