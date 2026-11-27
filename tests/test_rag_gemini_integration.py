import os
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from backend.bot import processar_mensagem_produtor
import llm_adapter

db_manager_module = sys.modules["db_manager"]


HISTORICO_MILHO = [
    {
        "direcao": "entrada",
        "tipo_mensagem": "texto_livre",
        "mensagem_texto": "Minha horta de alface está com pulgões.",
    },
    {
        "direcao": "saida",
        "tipo_mensagem": "resposta_llm",
        "mensagem_texto": "Inspecione a parte inferior das folhas.",
    },
    {
        "direcao": "entrada",
        "tipo_mensagem": "texto_livre",
        "mensagem_texto": "Minha plantação de milho está com folhas amareladas.",
    },
    {
        "direcao": "saida",
        "tipo_mensagem": "resposta_llm",
        "mensagem_texto": "As folhas mais velhas ou mais novas?",
    },
    {
        "direcao": "entrada",
        "tipo_mensagem": "texto_livre",
        "mensagem_texto": "As mais velhas.",
    },
    {
        "direcao": "saida",
        "tipo_mensagem": "resposta_llm",
        "mensagem_texto": "O amarelecimento é uniforme ou acontece entre as nervuras?",
    },
]


class RagGeminiIntegrationTests(unittest.TestCase):
    def test_profile_is_prompt_context_not_documentary_evidence(self):
        answer = "Observe as folhas inferiores, conforme o documento."
        profile = {
            "cultura_ativa": "citros",
            "tipo_solo_ativo": "argiloso",
            "area_ha_ativa": 2.5,
        }
        chunks = [
            {
                "documento": "Boletim 100",
                "pagina": 110,
                "texto": "A deficiência de fósforo pode alterar a coloração das folhas.",
            }
        ]
        with patch(
            "llm_adapter._chamar_gemini_api",
            return_value=answer,
        ) as gemini, patch(
            "llm_adapter._carregar_recursos_rag",
            return_value=(chunks, None, None, None, None),
        ), patch(
            "rag_retrieval.hybrid_search",
            return_value=[{"indice": 0}],
        ), patch(
            "rag_retrieval.rerank_candidates",
            return_value={
                "fora_do_escopo": False,
                "candidatos_reranqueados": [{"indice": 0}],
            },
        ), patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value=profile,
        ), patch(
            "bot.db_manager.salvar_mensagem_telegram",
        ), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=[],
        ):
            result = processar_mensagem_produtor(
                "test-chat",
                "Quais sintomas devo observar na deficiência de fósforo?",
            )

        self.assertEqual(result, answer)
        prompt = gemini.call_args.args[0]
        profile_context = prompt.split(
            "=== PERFIL DA LAVOURA (contexto cadastrado, não evidência técnica) ===\n",
            1,
        )[1].split("\n\n=== TRECHOS DOCUMENTAIS ===", 1)[0]
        self.assertIn("Cultura: citros", profile_context)
        self.assertIn("Tipo de solo: argiloso", profile_context)
        self.assertIn("Área: 2.5 ha", profile_context)
        self.assertIn("não é evidência técnica", prompt)
        document_context = prompt.split("=== TRECHOS DOCUMENTAIS ===\n", 1)[1]
        self.assertIn("Boletim 100; página: 110", document_context)
        self.assertNotIn("citros", document_context)
        self.assertNotIn("argiloso", document_context)
        self.assertNotIn("2.5 ha", document_context)

    def test_procedural_follow_up_prefers_latest_assistant_offer(self):
        older_offer = (
            "Quer que eu te explique como coletar folhas para análise foliar?"
        )
        recent_offer = (
            "A análise de solo ajuda a confirmar a disponibilidade dos nutrientes. "
            + ("Detalhes sobre a amostragem. " * 35)
            + "Quer que eu te explique como coletar essa amostra de solo?"
        )
        history = [
            {
                "direcao": "entrada",
                "tipo_mensagem": "texto_livre",
                "mensagem_texto": "Quero verificar a deficiência de fósforo.",
            },
            {
                "direcao": "saida",
                "tipo_mensagem": "resposta_llm",
                "mensagem_texto": older_offer,
            },
            {
                "direcao": "entrada",
                "tipo_mensagem": "texto_livre",
                "mensagem_texto": "As folhas mais velhas estão amarelas.",
            },
            {
                "direcao": "saida",
                "tipo_mensagem": "resposta_llm",
                "mensagem_texto": recent_offer,
            },
        ]

        for question in ("Como faço?", "Como coleto?", "Pode explicar?"):
            with self.subTest(question=question):
                selected = llm_adapter._historico_relevante(question, history)

                self.assertEqual(len(selected), 2)
                self.assertEqual(
                    selected[0]["texto"],
                    "As folhas mais velhas estão amarelas.",
                )
                self.assertTrue(
                    selected[1]["texto"].endswith(
                        "Quer que eu te explique como coletar essa amostra de solo?"
                    )
                )
                self.assertNotIn(
                    older_offer,
                    "\n".join(item["texto"] for item in selected),
                )

    def test_self_contained_question_starting_with_e_is_independent(self):
        question = (
            "E se minha planta apresentar deficiência de fósforo? "
            "Quais sintomas devo observar?"
        )

        self.assertFalse(llm_adapter._pergunta_depende_de_contexto(question))
        self.assertEqual(
            llm_adapter._historico_relevante(question, HISTORICO_MILHO),
            [],
        )

    def test_self_contained_e_question_retrieves_without_history_rewrite(self):
        question = (
            "E se minha planta apresentar deficiência de fósforo? "
            "Quais sintomas devo observar?"
        )
        answer = "Observe a coloração das folhas, conforme o documento."
        chunks = [
            {
                "documento": "Boletim 100",
                "pagina": 110,
                "texto": "A deficiência de fósforo pode alterar a coloração das folhas.",
            }
        ]
        with patch(
            "llm_adapter._chamar_gemini_api",
            return_value=answer,
        ) as gemini, patch(
            "llm_adapter._carregar_recursos_rag",
            return_value=(chunks, None, None, None, None),
        ), patch(
            "rag_retrieval.hybrid_search",
            return_value=[{"indice": 0}],
        ) as retrieval, patch(
            "rag_retrieval.rerank_candidates",
            return_value={
                "fora_do_escopo": False,
                "candidatos_reranqueados": [{"indice": 0}],
            },
        ), patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch(
            "bot.db_manager.salvar_mensagem_telegram",
        ), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=HISTORICO_MILHO,
        ):
            result = processar_mensagem_produtor("test-chat", question)

        self.assertEqual(result, answer)
        gemini.assert_called_once()
        self.assertEqual(retrieval.call_args.args[0], question)
        self.assertIn(question, gemini.call_args.args[0])
        self.assertIn(
            "Nenhum histórico necessário para esta pergunta.",
            gemini.call_args.args[0],
        )

    def test_context_dependent_phrasings_are_recognized(self):
        for question in (
            "E nesse caso?",
            "Como corrigir isso?",
            "Qual seria a causa?",
        ):
            with self.subTest(question=question):
                self.assertTrue(
                    llm_adapter._pergunta_depende_de_contexto(question)
                )

    def test_invalid_rewrite_json_returns_safe_clarification(self):
        responses = (
            '```json\n{"consulta_rag": "milho e sintomas", '
            '"precisa_esclarecimento": false, '
            '"pergunta_esclarecimento": ""}\n```',
            '{"consulta_rag": "milho e sintomas", '
            '"precisa_esclarecimento": "false", '
            '"pergunta_esclarecimento": ""}',
        )

        for response in responses:
            with self.subTest(response=response), patch(
                "llm_adapter._chamar_gemini_api",
                return_value=response,
            ):
                consulta, esclarecimento = (
                    llm_adapter._reformular_consulta_com_historico(
                        "Como corrigir isso?",
                        "Produtor: Meu milho está com folhas amareladas.",
                        "dummy-api-key",
                    )
                )

                self.assertIsNone(consulta)
                self.assertIn("repetir a pergunta completa", esclarecimento)

    def test_rewriter_request_enforces_json_schema(self):
        generated_json = (
            '{"consulta_rag":"Deficiência de fósforo em plantas e sintomas",'
            '"precisa_esclarecimento":false,"pergunta_esclarecimento":""}'
        )
        response = unittest.mock.Mock()
        response.status_code = 200
        response.json.return_value = {
            "candidates": [
                {
                    "content": {"parts": [{"text": generated_json}]},
                    "finishReason": "STOP",
                }
            ]
        }

        with patch(
            "llm_adapter.requests.post",
            return_value=response,
        ) as post:
            result = llm_adapter._chamar_gemini_api(
                "prompt de teste",
                "dummy-api-key",
                response_schema=llm_adapter.QUERY_REWRITE_RESPONSE_SCHEMA,
            )

        self.assertEqual(result, generated_json)
        generation_config = post.call_args.kwargs["json"]["generationConfig"]
        self.assertEqual(
            generation_config["responseMimeType"],
            "application/json",
        )
        self.assertEqual(
            generation_config["responseSchema"],
            llm_adapter.QUERY_REWRITE_RESPONSE_SCHEMA,
        )

    def test_answerable_question_sends_sources_to_existing_gemini_call(self):
        answer = "N, P e K apresentam alta redistribuição na planta."
        with patch(
            "llm_adapter._chamar_gemini_api",
            return_value=answer,
        ) as gemini, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=[],
        ):
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

    def test_follow_up_uses_relevant_history_to_rewrite_retrieval_query(self):
        rewrite = (
            '{"consulta_rag": "Milho com amarelecimento uniforme nas folhas '
            'mais velhas", "precisa_esclarecimento": false, '
            '"pergunta_esclarecimento": ""}'
        )
        answer = "O documento descreve a redistribuição dos nutrientes."
        chunks = [
            {
                "documento": "Boletim 100",
                "pagina": 201,
                "texto": "Teores de N, P e K diminuem com a idade da folha.",
            }
        ]
        with patch(
            "llm_adapter._chamar_gemini_api",
            side_effect=[rewrite, answer],
        ) as gemini, patch(
            "llm_adapter._carregar_recursos_rag",
            return_value=(chunks, None, None, None, None),
        ), patch(
            "rag_retrieval.hybrid_search",
            return_value=[{"indice": 0}],
        ) as retrieval, patch(
            "rag_retrieval.rerank_candidates",
            return_value={
                "fora_do_escopo": False,
                "candidatos_reranqueados": [{"indice": 0}],
            },
        ) as reranker, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch(
            "bot.db_manager.salvar_mensagem_telegram",
            return_value=100,
        ), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=HISTORICO_MILHO,
        ):
            response = processar_mensagem_produtor(
                "test-chat",
                "É uniforme.",
            )

        self.assertEqual(response, answer)
        self.assertEqual(gemini.call_count, 2)
        self.assertEqual(
            retrieval.call_args.args[0],
            "Milho com amarelecimento uniforme nas folhas mais velhas",
        )
        self.assertEqual(
            reranker.call_args.args[0],
            "Milho com amarelecimento uniforme nas folhas mais velhas",
        )
        rewrite_prompt = gemini.call_args_list[0].args[0]
        final_prompt = gemini.call_args_list[1].args[0]
        self.assertEqual(
            gemini.call_args_list[0].kwargs["response_schema"],
            llm_adapter.QUERY_REWRITE_RESPONSE_SCHEMA,
        )
        self.assertIn("Minha plantação de milho", rewrite_prompt)
        self.assertNotIn("alface", rewrite_prompt)
        self.assertIn("É uniforme.", final_prompt)
        self.assertIn("=== PERGUNTA ORIGINAL DO PRODUTOR ===\nÉ uniforme.", final_prompt)
        self.assertIn("Produtor: As mais velhas.", final_prompt)
        self.assertIn("não é fonte de fatos técnicos", final_prompt)
        self.assertIn("Boletim 100; página: 201", final_prompt)

    def test_ambiguous_follow_up_asks_for_clarification_before_retrieval(self):
        rewrite = (
            '{"consulta_rag": "", "precisa_esclarecimento": true, '
            '"pergunta_esclarecimento": "Qual planta e qual sintoma você observou?"}'
        )
        history = [
            {
                "direcao": "entrada",
                "tipo_mensagem": "texto_livre",
                "mensagem_texto": "Minha planta está com algum problema.",
            },
            {
                "direcao": "saida",
                "tipo_mensagem": "resposta_llm",
                "mensagem_texto": "Pode descrever os sintomas?",
            },
        ]
        with patch(
            "llm_adapter._chamar_gemini_api",
            return_value=rewrite,
        ) as gemini, patch(
            "rag_retrieval.hybrid_search"
        ) as retrieval, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=history,
        ):
            response = processar_mensagem_produtor("test-chat", "Isso.")

        self.assertEqual(
            response,
            "Qual planta e qual sintoma você observou?",
        )
        gemini.assert_called_once()
        retrieval.assert_not_called()

    def test_context_dependent_message_without_history_is_not_searched_alone(self):
        with patch("llm_adapter._chamar_gemini_api") as gemini, patch(
            "rag_retrieval.hybrid_search"
        ) as retrieval, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=[],
        ):
            response = processar_mensagem_produtor(
                "test-chat",
                "As mais velhas.",
            )

        self.assertIn("repetir a pergunta completa", response)
        gemini.assert_not_called()
        retrieval.assert_not_called()

    def test_database_returns_prior_messages_in_chronological_order(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "messages.sqlite"
            with closing(sqlite3.connect(database_path)) as connection:
                with connection:
                    connection.execute(
                        """
                        CREATE TABLE mensagens_telegram (
                            id INTEGER PRIMARY KEY,
                            chat_id TEXT NOT NULL,
                            direcao TEXT NOT NULL,
                            tipo_mensagem TEXT,
                            mensagem_texto TEXT NOT NULL,
                            criado_em TEXT
                        )
                        """
                    )
                    connection.executemany(
                        """
                        INSERT INTO mensagens_telegram
                            (id, chat_id, direcao, tipo_mensagem, mensagem_texto)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        [
                            (1, "chat-a", "entrada", "texto_livre", "Primeira"),
                            (2, "chat-a", "saida", "resposta_llm", "Segunda"),
                            (3, "chat-a", "entrada", "texto_livre", "Atual"),
                            (4, "chat-b", "entrada", "texto_livre", "Outro chat"),
                        ],
                    )

            with patch.object(
                db_manager_module,
                "LOCAL_DB_PATH",
                str(database_path),
            ):
                history = db_manager_module.db_manager.listar_mensagens_telegram(
                    "chat-a",
                    limit=8,
                    antes_de_id=3,
                )

        self.assertEqual([message["id"] for message in history], [1, 2])
        self.assertEqual(
            [message["mensagem_texto"] for message in history],
            ["Primeira", "Segunda"],
        )

    def test_unanswerable_question_does_not_call_gemini(self):
        with patch("llm_adapter._chamar_gemini_api") as gemini, patch(
            "bot.db_manager.obter_ou_criar_produtor",
            return_value={},
        ), patch("bot.db_manager.salvar_mensagem_telegram"), patch(
            "bot.db_manager.listar_mensagens_telegram",
            return_value=[],
        ):
            response = processar_mensagem_produtor(
                "test-chat",
                "Como combater uma praga que não aparece nos documentos?",
            )

        gemini.assert_not_called()
        self.assertIn("Não encontrei informação suficiente", response)


if __name__ == "__main__":
    unittest.main()
