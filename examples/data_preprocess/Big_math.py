
from modelscope.msdatasets import MsDataset


import argparse
import os
import re

import datasets

# from verl.utils.hdfs_io import copy, makedirs

# @misc{albalak2025bigmathlargescalehighqualitymath,
#       title={Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models}, 
#       author={Alon Albalak and Duy Phung and Nathan Lile and Rafael Rafailov and Kanishk Gandhi and Louis Castricato and Anikait Singh and Chase Blagden and Violet Xiang and Dakota Mahan and Nick Haber},
#       year={2025},
#       eprint={2502.17387},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG},
#       url={https://arxiv.org/abs/2502.17387}, 
# }



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/model/data/Big-Math-RL-Verified")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "SynthLabsAI/Big-Math-RL-Verified"

    ds =  MsDataset.load('SynthLabsAI/Big-Math-RL-Verified', subset_name='default', split='train')

    instruction_following = "Let's think step by step and output the final answer in \\boxed{}."
 
    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("problem")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            answer_match = re.search(r"\\boxed\{(.*?)\}", answer_raw)
            if answer_match:
                print("WARNING: answer already has \\boxed{}" + answer_raw )
                ground_truth_boxed = answer_raw
            else:
                ground_truth_boxed = "\\boxed{" + answer_raw + "}"
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": ground_truth_boxed},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = ds.map(function=make_map_fn("train"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

    # if hdfs_dir is not None:
    #     makedirs(hdfs_dir)
    #     copy(src=local_dir, dst=hdfs_dir)
