'''script to run inference on the model fine-tuned with ft_llama2.py'''
import pandas as pd

from ludwig.api import LudwigModel

MODELDIR = "results/api_experiment_run/model/"

def main():
    '''main function for inference with Llama2'''


    model = LudwigModel.load(MODELDIR)

    test_examples = pd.DataFrame(
        [
            {
                "question": "What structure is classified as a definite lie algebra?",
            },
            {
                "question": "What type of laser is used to study infrared?",
            },
            {
                "question": "What type of detector detects photon arrival?",
            },
            {
                "question": "Can a qw be made shapeless?",
            },
            {
                "question": "Which of the following is the only finite width of quark gluon\
                plasma bags?",
            },
            {
                "question": "Which phase is associated with a hexagon of hexagons diffraction?",
            },
            {
                "question": "Where is the companion galaxy?",
            },
        ]
    )

    predictions, _ = model.predict(dataset=test_examples)
    print(predictions)

    for input_with_prediction in zip(
        test_examples["question"], predictions["answer_response"]
    ):
        print(f"Instruction: {input_with_prediction[0]}")
        print(f"Generated Output: {input_with_prediction[1][0]}")
        print("\n\n")

    return 0


if __name__ == "__main__":
    main()
