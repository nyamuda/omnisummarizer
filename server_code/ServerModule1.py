import anvil.secrets
import anvil.server
from huggingface_hub import InferenceClient
from huggingface_hub.inference_api import InferenceApi
import re
API_TOKEN=anvil.secrets.get_secret('HUGGINGFACE_API_TOKEN')
client = InferenceClient("nyamuda/extractive-summarization",
                         token=API_TOKEN)

# Initialize the Inference API
inference = InferenceApi(
    repo_id="nyamuda/extractive-summarization", token=API_TOKEN, task="summarization")

# This is a server module. It runs on the Anvil server,
# rather than in the user's browser.
#
# To allow anvil.server.call() to call functions here, we mark
# them with @anvil.server.callable.
# Here is an example - you can replace it with your own:

@anvil.server.callable
def clean_summary(text):
    # Fix spacing before punctuation
    text = re.sub(r'\s+([.,?!])', r'\1', text)

    # Capitalize first letter of each sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.capitalize() for s in sentences]

    # Join the sentences back together
    return " ".join(sentences)
  
@anvil.server.callable
def summarize_text(text):
  prompt=f"summarize: {text}"
  response=client.summarization(prompt)
  #clean summary
  summary=clean_summary(response["summary_text"])
  return summary
#summarize using the inference API
#this allows to set the max and min summary length
@anvil.server.callable
def summarize_with_length(prompt,text, max_length,min_length):
  parameters = {
        "max_length": max_length,
        "min_length": min_length,
        "num_beams": 4,  # Optional: adjust for better generation quality
        "length_penalty": 2.0  # Optional: adjust penalty to balance length
    }
  response = inference(f"{prompt}: {text}", params=parameters)
  # Get the summarized text
  summaryText = response[0]["summary_text"]
  return summaryText
  

@anvil.server.route("/summarize")
def summary_request():
  prompt = anvil.server.request.body_json["prompt"]
  text = anvil.server.request.body_json["text"]
  max_length = anvil.server.request.body_json["max_length"]
  min_length = anvil.server.request.body_json["min_length"]

  summaryText=summarize_with_length(prompt,text,max_length,min_length)
  
  return {"summary":summaryText}

@anvil.server.callable
def test_code():
    text="Johannes Gutenberg (1398 – 1468) was a German goldsmith and publisher who introduced printing to Europe. His introduction of mechanical movable type printing to Europe started the Printing Revolution and is widely regarded as the most important event of the modern period."
    prompt="summarize"
    max_length=512
    min_length=5

    summaryText=summarize_with_length(prompt,text,max_length,min_length)
    print(summaryText)


test_code()



