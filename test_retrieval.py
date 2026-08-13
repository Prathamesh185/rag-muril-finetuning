from rag.retriever import retrieve


question = "यूरिया में कितना नाइट्रोजन होता है?"

results = retrieve(
    question,
    top_k=5
)

print("\nQuestion:")
print(question)

for result in results:

    print("\n" + "=" * 60)

    print("Rank :", result["rank"])
    print("Score:", round(result["score"], 4))
    print("Title:", result["title"])
    print("Text :", result["text"][:500])