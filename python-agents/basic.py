from ollama import chat
from ollama import ChatResponse
from IPython.display import display, Markdown

response: ChatResponse = chat(model='gpt-oss:20b', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
display(Markdown(response.message.content))
# or access fields directly from the response object
print(response.message.content)