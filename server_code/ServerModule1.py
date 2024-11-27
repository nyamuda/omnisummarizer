import anvil.secrets
import anvil.server
from huggingface_hub import InferenceClient
import re
API_TOKEN=anvil.secrets.get_secret('HUGGINGFACE_API_TOKEN')
headers = {"max_length": 50}
client = InferenceClient("nyamuda/extractive-summarization",
                         token=API_TOKEN)

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


@anvil.server.route("/summarize")
def summary_request():
  text = anvil.server.request.body_json["text"]
  response=summarize_text(text)
  return {"summary":response}
