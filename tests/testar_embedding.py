import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer


def main():
    modelo = SentenceTransformer("BAAI/bge-m3", local_files_only=True)

    textos = [
        "Como identificar deficiência de nitrogênio no milho?",
        "A falta de nitrogênio causa amarelecimento nas folhas."
    ]

    embeddings = modelo.encode(textos, normalize_embeddings=True)

    print("Formato:", embeddings.shape)
    print("Modelo carregado com sucesso!")


if __name__ == "__main__":
    main()