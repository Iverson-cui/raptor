import os
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import UnifiedQAModel, DeepSeekQAModel

print("reading sample.txt from demo directory")
with open("demo/sample.txt", "r") as file:
    text = file.read()

print(text[:100])

print("Creating RetrievalAugmentation instance")
# RA = RetrievalAugmentation(
#     config=RetrievalAugmentationConfig(qa_model=DeepSeekQAModel())
# )
# RA = RetrievalAugmentation()
RA = RetrievalAugmentation(
    config=RetrievalAugmentationConfig(qa_model=UnifiedQAModel())
)

print("Adding document to RA instance")
# construct the tree and corresponding retriever
RA.add_documents(text)

print("Performing retrieval-augmented generation")
question = "What is this story about? Summarize it in a few sentences."

answer = RA.answer_question(question=question)

print("Answer: ", answer)

print("Save the tree.")
# Save the tree by calling RA.save("path/to/save")
SAVE_PATH = "demo/cinderella_myself"
RA.save(SAVE_PATH)

# print("Load the tree.")
# RA = RetrievalAugmentation(tree=SAVE_PATH)
# answer = RA.answer_question(question=question)
# print("Answer: ", answer)
