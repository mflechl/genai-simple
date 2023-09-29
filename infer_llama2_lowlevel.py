"""for low-level inference access to llama2. Not complete yet"""

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "results/api_experiment_run/model/model_weights/"


test_examples = [
    "What structure is classified as a definite lie algebra?",
    "What type of laser is used to study infrared?",
    "What type of detector detects photon arrival?",
    "Can a qw be made shapeless?",
    "Which of the following is the only finite width of quark gluon plasma bags?",
    "Which phase is associated with a hexagon of hexagons diffraction?",
    "Where is the companion galaxy?",
]

print(test_examples)

config = PeftConfig.from_pretrained("results/api_experiment_run/model/model_weights/")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(
    model, "results/api_experiment_run/model/model_weights/"
)
# 1 tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-hf", padding_side="left"
)
tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# 1 token_ids = [torch.squeeze(tokenizer(ex,return_tensors='pt',truncation=True)['input_ids'],0) \
#    for ex in test_examples]
# 1 token_ids = pad_sequence(token_ids, batch_first=True, padding_value=-1)

token_ids = tokenizer(test_examples, return_tensors="pt", padding=True, truncation=True)
print(token_ids["input_ids"])
print(token_ids["attention_mask"])

with torch.no_grad():
    gen_tokens = model.generate(
        **token_ids,
        do_sample=False,
        max_new_tokens=50,
    )
gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
print(gen_text)


# predictions = model.predict(test_examples)[0]

# for input_with_prediction in zip(test_examples['question'], predictions['answer_response']):
#  print(f"Instruction: {input_with_prediction[0]}")
#  print(f"Generated Output: {input_with_prediction[1][0]}")
#  print("\n\n")
