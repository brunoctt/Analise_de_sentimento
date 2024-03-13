from classification.evaluation import classify_phrase


def main():
    phrase = ""
    print("Escreva uma frase, ou 0 para sair")
    phrase = input()
    while phrase != "0":
        result = classify_phrase(phrase)
        print("Sentimento:", result)
        print("Escreva uma frase, ou 0 para sair")
        phrase = input()


if __name__ == "__main__":
    main()
