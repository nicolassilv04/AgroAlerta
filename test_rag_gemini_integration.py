import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.bot import processar_mensagem_produtor


class RagGeminiIntegrationTests(unittest.TestCase):
    def test_answerable_question_sends_sources_to_existing_gemini_call(self):
        answer = "N, P e K apresentam alta redistribuição na planta."
        with patch(
            "llm_adapter._chamar_gemini_api",
            return_value=answer,
        ) as gemini, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"):
            response = processar_mensagem_produtor(
                "test-chat",
                "Quais nutrientes apresentam alta mobilidade na planta?",
            )

        self.assertEqual(response, answer)
        gemini.assert_called_once()
        prompt = gemini.call_args.args[0]
        self.assertIn("Boletim 100", prompt)
        document_context = prompt.split("=== TRECHOS DOCUMENTAIS ===\n", 1)[1]
        self.assertTrue(
            document_context.startswith(
                "[Documento: Boletim 100; página: 201]"
            )
        )
        self.assertIn("alta redistribuição", prompt.casefold())
        self.assertIn("exclusivamente", prompt)
        self.assertIn("não complete lacunas", prompt)

    def test_unanswerable_question_does_not_call_gemini(self):
        with patch("llm_adapter._chamar_gemini_api") as gemini, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"):
            response = processar_mensagem_produtor(
                "test-chat",
                "Como combater uma praga que não aparece nos documentos?",
            )

        gemini.assert_not_called()
        self.assertIn("Não encontrei informação suficiente", response)


if __name__ == "__main__":
    unittest.main()
