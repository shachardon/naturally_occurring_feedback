import datasets

TOKEN = "Insert your HF Token Here"
LOADING_MODE = 1

extraction_df = datasets.load_dataset("shachardon/naturally_occurring_feedback")["train"]
lmsys_dataset = datasets.load_dataset("lmsys/lmsys-chat-1m", token=TOKEN)["train"]
lmsys_dataset = lmsys_dataset.to_pandas()

# use the conversation_id and feedback_turn from the extraction_df to get the conversations
# from the lmsys dataset
conversations = []
for i in range(len(extraction_df)):
    row = extraction_df[i]
    conversation_id = row["conversation_id"]
    feedback_turn = row["feedback_turn"]
    org_row = lmsys_dataset[lmsys_dataset["conversation_id"] == conversation_id]
    conversation = org_row["conversation"]

    conversation_context = []
    if LOADING_MODE == 1:
      for i in range(feedback_turn * 2 - 2, feedback_turn * 2):
          conversation_context.append(conversation[i])
    elif LOADING_MODE == 2:
      for i in range(feedback_turn * 2 - 2, feedback_turn * 2 + 1):
          conversation_context.append(conversation[i])
    else:
      conversation_context = conversation

    conversations.append({"conversation_id": conversation_id, "model": org_row["model"],
                          "conversation_context": conversation_context, "feedback_turn": feedback_turn,
                          "language": org_row["language"], "category": row["category"], "label": row["label"]})
print(conversations[:5])
