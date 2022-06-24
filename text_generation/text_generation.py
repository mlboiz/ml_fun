from transformers import TFAutoModelWithLMHead, AutoTokenizer


class TextGenerator:
    def __init__(self, max_length=100):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = TFAutoModelWithLMHead.from_pretrained(
            "gpt2"
        )
        self.max_length = max_length

    def generate(self, input_context=""):
        input_ids = None if input_context == "" else self.tokenizer.encode(input_context, return_tensors="tf")
        outputs = self.model.generate(
            input_ids=input_ids, max_length=100,
            do_sample=None if input_ids is None else True,
        )
        return f"Generated: \n{self.tokenizer.decode(outputs[0], skip_special_tokens=True)}"
