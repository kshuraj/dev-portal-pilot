"""
Hello World:
A small program to demonstrate how the app package is supposed to be organised.
Here we've used 'gpt2' model from HuggingFace to generate text for user's
input.
"""
from transformers import pipeline

def run_model(text):
    """
    Example taken from: https://huggingface.co/gpt2#how-to-use
    """
    generator = pipeline('text-generation', model='gpt2')
    output = generator(text)
    return output[0].get('generated_text')


def main(text):
    """
    Argument(s) name in the function signature are matching with the id in the
    app.config
    """
    output = run_model(text)
    return output