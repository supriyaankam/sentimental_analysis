from sentiment import analyze_sentiment

text = input("Enter a sentence: ")

result = analyze_sentiment(text)

print("\n--- Sentiment Result ---")
print(result)