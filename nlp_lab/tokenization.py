import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def ensure_nltk_data() -> None:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('brown', quiet=True)


def main() -> None:
    ensure_nltk_data()

    text = """Natural Language Processing (NLP) is a field of AI that helps computers understand human language.
Python makes it easy to work with NLP thanks to libraries like NLTK and TextBlob."""

    sentences = sent_tokenize(text)
    print("Sentences:")
    for sentence in sentences:
        print("-", sentence)

    words = word_tokenize(text)
    print("\nWords:")
    print(words)

    demo_text = "I don't like this 😕, but NLP is fun!"
    print("\nDemo text:")
    print(demo_text)
    print("Tokens:")
    print(word_tokenize(demo_text))


if __name__ == "__main__":
    main()
