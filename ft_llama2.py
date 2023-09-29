'''script to finetune Llama2'''
#import getpass
import locale
import logging
import os
#import yaml
import json

import torch
import numpy as np
import pandas as pd

from ludwig.api import LudwigModel

locale.getpreferredencoding = lambda: "UTF-8"
np.random.seed(123)


def clear_cache():
    '''clears the cuda cache'''
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main(n_examples=500):
    '''main function to run run fine-tuning'''

    # os.environ["HUGGING_FACE_HUB_TOKEN"] = getpass.getpass("Token:")
    assert os.environ["HUGGING_FACE_HUB_TOKEN"]

    with open("arxiv_physics_instruct_30k.jsonl", encoding="utf-8") as f_in:
        data1 = [json.loads(line) for line in f_in]
    with open("arxiv_math_instruct_50k.jsonl", encoding="utf-8") as f_in:
        data2 = [json.loads(line) for line in f_in]

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df_both = pd.concat([df1, df2])
    main_df = df_both.sample(frac=1, random_state=42)
    main_df.reset_index(drop=True, inplace=True)

    # We're going to create a new column called `split` where:
    # 90% will be assigned a value of 0 -> train set
    # 5% will be assigned a value of 1 -> validation set
    # 5% will be assigned a value of 2 -> test set
    # Calculate the number of rows for each split value
    total_rows = len(main_df)
    split_0_count = int(total_rows * 0.9)
    split_1_count = int(total_rows * 0.05)
    split_2_count = total_rows - split_0_count - split_1_count

    # Create an array with split values based on the counts
    split_values = np.concatenate(
        [np.zeros(split_0_count), np.ones(split_1_count), np.full(split_2_count, 2)]
    )

    # Shuffle the array to ensure randomness
    np.random.shuffle(split_values)

    # Add the 'split' column to the DataFrame
    main_df["split"] = split_values
    main_df["split"] = main_df["split"].astype(int)

    # We will just use a small fraction of this dataset.
    main_df = main_df.head(n=n_examples)

    print(main_df.head(3))
    print(f"Total number of examples in the dataset: {main_df.shape[0]}")

    # Calculating the length of each cell in each column
    #df["num_characters_question"] = df["question"].apply(lambda x: len(x))
    #df["num_characters_answer"] = df["answer"].apply(lambda x: len(x))

    main_df["num_characters_question"] = main_df["question"].apply(len)
    main_df["num_characters_answer"] = main_df["answer"].apply(len)

    # Show Distribution
    main_df.hist(column=["num_characters_question", "num_characters_answer"])

    # Calculating the average
    average_chars_instruction = main_df["num_characters_question"].mean()
    # average_chars_input = main_df['num_characters_input'].mean()
    average_chars_output = main_df["num_characters_answer"].mean()

    print(
        f"Average number of tokens in the instruction column: {(average_chars_instruction / 3):.0f}"
    )
    # print(f'Average number of tokens in the input column: {(average_chars_input / 3):.0f}')
    print(
        f"Average number of tokens in the output column: {(average_chars_output / 3):.0f}",
        end="\n\n",
    )


    clear_cache()

    # Using Llama-2, 5 epochs
    model = LudwigModel(config="config_ft.yaml", logging_level=logging.INFO)
#    results = model.train(dataset=main_df)
    model.train(dataset=main_df)

    return 0


if __name__ == "__main__":
    main()
