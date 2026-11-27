import pymupdf
import json

CAMINHO_PDF = "documentos/boletim_100.pdf"

TAMANHO_CHUNK = 500  # palavras
OVERLAP = 50         # palavras repetidas

pdf = pymupdf.open(CAMINHO_PDF)

chunks = []

for numero, pagina in enumerate(pdf):
    texto = pagina.get_text().strip()

    if not texto:
        continue

    palavras = texto.split()
    inicio = 0

    while inicio < len(palavras):
        fim = inicio + TAMANHO_CHUNK

        chunk_texto = " ".join(palavras[inicio:fim])

        chunks.append({
            "documento": "Boletim 100",
            "pagina": numero + 1,
            "texto": chunk_texto
        })

        inicio += TAMANHO_CHUNK - OVERLAP

for numero in [135, 136]:
    print(f"\n--- PÁGINA {numero + 1} ---")
    print(pdf[numero].get_text())

with open("documentos/chunks.json", "w", encoding="utf-8") as arquivo:
    json.dump(chunks, arquivo, ensure_ascii=False, indent=4)

print("\nChunks salvos em documentos/chunks.json")

print("Total de chunks:", len(chunks))

print("\nPrimeiro chunk:")
print(chunks[0])

print("\nÚltimo chunk:")
print(chunks[-1])