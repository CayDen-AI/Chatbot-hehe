from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline


def get_model(model_name, max_new_tokens=512):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        trust_remote_code=True,
        device_map='auto'
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

