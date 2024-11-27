import anvil.secrets
import anvil.server
from huggingface_hub import InferenceClient
API_TOKEN=anvil.secrets.get_secret('HUGGINGFACE_API_TOKEN')

client = InferenceClient(
    "nyamuda/extractive-summarization",
    token=API_TOKEN,
)

# This is a server module. It runs on the Anvil server,
# rather than in the user's browser.
#
# To allow anvil.server.call() to call functions here, we mark
# them with @anvil.server.callable.
# Here is an example - you can replace it with your own:
@anvil.server.callable
def say_hello(text):
  prompt=f"summarize: {text}"
  response=client.summarization(prompt)
  return response


@anvil.server.route("/summarize")
def summarize():
  text = anvil.server.request.body_json["text"]
  response=say_hello(text)
  return {"output":response}
