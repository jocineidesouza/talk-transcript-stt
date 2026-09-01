import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import ANY, AsyncMock, MagicMock, patch


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = ROOT_DIR / "stt" / "app.py"

STT_APP = None
STT_APP_IMPORT_ERROR = None
try:
    spec = importlib.util.spec_from_file_location("stt_app_under_test", APP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"nao foi possivel carregar modulo: {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    STT_APP = module
except Exception as exc:  # pragma: no cover - depends on local environment
    STT_APP_IMPORT_ERROR = exc


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class NamespaceParsingTests(unittest.TestCase):
    def test_extract_room_namespace_accepts_all_allowed_namespaces(self):
        for namespace in sorted(STT_APP.ALLOWED_LIVEKIT_NAMESPACES):
            room_name = f"{namespace}__abc123"
            info = STT_APP.extract_room_namespace(room_name)
            self.assertIsNotNone(info)
            self.assertEqual(info.namespace, namespace)
            self.assertEqual(info.room_id, "abc123")

    def test_extract_room_namespace_rejects_invalid_formats(self):
        invalid_rooms = [
            "talk__dev",
            "talk__qa__abc123",
            "ellevo-connect__dev__",
            "invalid-room-format",
        ]
        for room_name in invalid_rooms:
            self.assertIsNone(STT_APP.extract_room_namespace(room_name))


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class FirebaseRouterTests(unittest.TestCase):
    def build_routing(self) -> object:
        return STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path=(
                "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1"
            ),
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )

    def test_router_reuses_cached_sink_for_same_namespace(self):
        conf = STT_APP.FirebaseNamespaceConfig(
            project_id="talk-dev",
            storage_bucket="talk-dev.appspot.com",
            credentials_file="/secrets/talk-dev-firebaseadmin.json",
        )
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={"talk__dev": conf})

        fake_app = object()
        fake_fs_client = object()
        fake_bucket = object()

        with patch.object(STT_APP.firebase_admin, "get_app", side_effect=ValueError("missing")):
            with patch.object(STT_APP.credentials, "Certificate", return_value=object()):
                with patch.object(
                    STT_APP.firebase_admin, "initialize_app", return_value=fake_app
                ) as init_app_mock:
                    with patch.object(
                        STT_APP.firestore, "client", return_value=fake_fs_client
                    ) as fs_client_mock:
                        with patch.object(
                            STT_APP.storage, "bucket", return_value=fake_bucket
                        ) as bucket_mock:
                            sink_1 = router.sink_for_room("talk__dev__roomA")
                            sink_2 = router.sink_for_room("talk__dev__roomB")

        self.assertIs(sink_1, sink_2)
        self.assertTrue(sink_1.enabled)
        self.assertEqual(sink_1.namespace, "talk__dev")
        self.assertEqual(init_app_mock.call_count, 1)
        self.assertEqual(fs_client_mock.call_count, 1)
        self.assertEqual(bucket_mock.call_count, 1)

    def test_router_returns_disabled_sink_when_namespace_has_no_config(self):
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={})
        sink = router.sink_for_room("talk__dev__roomA")
        self.assertFalse(sink.enabled)
        self.assertEqual(sink.namespace, "disabled")

    def test_router_returns_disabled_sink_when_room_namespace_is_invalid(self):
        router = STT_APP.FirebaseRouter(
            enabled=True,
            configs_by_namespace={
                "talk__dev": STT_APP.FirebaseNamespaceConfig(
                    project_id="talk-dev",
                    storage_bucket="talk-dev.appspot.com",
                    credentials_file="/secrets/talk-dev-firebaseadmin.json",
                )
            },
        )
        sink = router.sink_for_room("invalid__room")
        self.assertFalse(sink.enabled)
        self.assertEqual(sink.namespace, "disabled")

    def test_router_wrappers_use_disabled_sink_without_fallback(self):
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={})
        disabled_sink = router.sink_for_room("talk__dev__roomA")
        disabled_sink.publish_call_index = MagicMock()
        routing = self.build_routing()

        with patch.object(router, "sink_for_namespace", return_value=disabled_sink):
            router.publish_call_index(
                routing=routing,
                status="processing",
                last_minute_index=3,
                finalized=False,
            )

        disabled_sink.publish_call_index.assert_called_once()
        args, _kwargs = disabled_sink.publish_call_index.call_args
        self.assertEqual(args[0], routing)
        self.assertEqual(args[1], "processing")
        self.assertEqual(args[2], 3)
        self.assertFalse(args[3])

    def test_resolve_room_routing_context_success(self):
        conf = STT_APP.FirebaseNamespaceConfig(
            project_id="talk-dev",
            storage_bucket="talk-dev.appspot.com",
            credentials_file="/secrets/talk-dev-firebaseadmin.json",
        )
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={"talk__dev": conf})

        fake_snapshot = MagicMock()
        fake_snapshot.exists = True
        fake_snapshot.to_dict.return_value = {"vertical": "HEALTH", "slug": "acme"}
        fake_doc = MagicMock()
        fake_doc.get.return_value = fake_snapshot
        fake_collection = MagicMock()
        fake_collection.document.return_value = fake_doc
        fake_fs_client = MagicMock()
        fake_fs_client.collection.return_value = fake_collection
        fake_sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        with patch.object(router, "_get_or_create_sink", return_value=fake_sink):
            routing = router.resolve_room_routing_context(
                "talk__dev__roomA", "RM_session-1", "session-1"
            )

        self.assertEqual(routing.namespace, "talk__dev")
        self.assertEqual(routing.room_id, "roomA")
        self.assertEqual(routing.vertical, "HEALTH")
        self.assertEqual(routing.slug, "acme")
        self.assertEqual(
            routing.firestore_doc_path,
            "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
        )
        self.assertEqual(
            routing.storage_base_path,
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )
        fake_fs_client.collection.assert_called_once_with("LIVEKIT_ROOM_INDEX")
        fake_collection.document.assert_called_once_with("talk__dev__roomA")

    def test_resolve_room_routing_context_not_found(self):
        conf = STT_APP.FirebaseNamespaceConfig(
            project_id="talk-dev",
            storage_bucket="talk-dev.appspot.com",
            credentials_file="/secrets/talk-dev-firebaseadmin.json",
        )
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={"talk__dev": conf})

        fake_snapshot = MagicMock()
        fake_snapshot.exists = False
        fake_doc = MagicMock()
        fake_doc.get.return_value = fake_snapshot
        fake_collection = MagicMock()
        fake_collection.document.return_value = fake_doc
        fake_fs_client = MagicMock()
        fake_fs_client.collection.return_value = fake_collection
        fake_sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        with patch.object(router, "_get_or_create_sink", return_value=fake_sink):
            with self.assertRaises(STT_APP.RoomRoutingError) as ctx:
                router.resolve_room_routing_context("talk__dev__roomA", "RM_session-1", None)

        self.assertEqual(ctx.exception.code, "room_index_not_found")

    def test_resolve_room_routing_context_invalid_payload(self):
        conf = STT_APP.FirebaseNamespaceConfig(
            project_id="talk-dev",
            storage_bucket="talk-dev.appspot.com",
            credentials_file="/secrets/talk-dev-firebaseadmin.json",
        )
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={"talk__dev": conf})

        fake_snapshot = MagicMock()
        fake_snapshot.exists = True
        fake_snapshot.to_dict.return_value = {"vertical": "", "slug": "acme"}
        fake_doc = MagicMock()
        fake_doc.get.return_value = fake_snapshot
        fake_collection = MagicMock()
        fake_collection.document.return_value = fake_doc
        fake_fs_client = MagicMock()
        fake_fs_client.collection.return_value = fake_collection
        fake_sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        with patch.object(router, "_get_or_create_sink", return_value=fake_sink):
            with self.assertRaises(STT_APP.RoomRoutingError) as ctx:
                router.resolve_room_routing_context("talk__dev__roomA", "RM_session-1", None)

        self.assertEqual(ctx.exception.code, "room_index_invalid")

    def test_fetch_agent_prompt_returns_prompt_when_document_exists(self):
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={})
        routing = self.build_routing()
        fake_snapshot = MagicMock()
        fake_snapshot.exists = True
        fake_snapshot.to_dict.return_value = {"prompt": "  Prompt do agente  "}
        fake_doc = MagicMock()
        fake_doc.get.return_value = fake_snapshot
        fake_fs_client = MagicMock()
        fake_fs_client.document.return_value = fake_doc
        fake_sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        with patch.object(router, "sink_for_namespace", return_value=fake_sink):
            prompt = router.fetch_agent_prompt(routing, "stt_summarize_minute")

        self.assertEqual(prompt, "Prompt do agente")
        fake_fs_client.document.assert_called_once_with(
            "VERTICALS/HEALTH/COMPANIES/acme/SETTINGS/ai_agents/AGENTS/stt_summarize_minute"
        )

    def test_fetch_agent_prompt_returns_none_when_document_does_not_exist(self):
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={})
        routing = self.build_routing()
        fake_snapshot = MagicMock()
        fake_snapshot.exists = False
        fake_doc = MagicMock()
        fake_doc.get.return_value = fake_snapshot
        fake_fs_client = MagicMock()
        fake_fs_client.document.return_value = fake_doc
        fake_sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        with patch.object(router, "sink_for_namespace", return_value=fake_sink):
            prompt = router.fetch_agent_prompt(routing, "stt_merge_summaries")

        self.assertIsNone(prompt)

    def test_fetch_agent_prompt_returns_none_when_prompt_is_invalid(self):
        router = STT_APP.FirebaseRouter(enabled=True, configs_by_namespace={})
        routing = self.build_routing()

        invalid_payloads = [
            {"prompt": 123},
            {"prompt": ""},
            {"prompt": "   "},
            {},
            None,
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                fake_snapshot = MagicMock()
                fake_snapshot.exists = True
                fake_snapshot.to_dict.return_value = payload
                fake_doc = MagicMock()
                fake_doc.get.return_value = fake_snapshot
                fake_fs_client = MagicMock()
                fake_fs_client.document.return_value = fake_doc
                fake_sink = STT_APP.FirebaseSink(
                    namespace="talk__dev",
                    enabled=True,
                    firestore_client=fake_fs_client,
                    storage_bucket=None,
                )

                with patch.object(router, "sink_for_namespace", return_value=fake_sink):
                    prompt = router.fetch_agent_prompt(routing, "stt_finalize_summary")

                self.assertIsNone(prompt)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryEnginePromptTests(unittest.TestCase):
    def build_engine(self) -> object:
        return STT_APP.SummaryEngine(
            enabled=True,
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            extra_headers={},
            minute_model="minute-model",
            accumulated_model="accumulated-model",
            final_model="final-model",
            final_text_model="final-text-model",
            timeout_seconds=20,
            request_retries=2,
            retry_base_seconds=1.5,
        )

    def test_summarize_minute_uses_external_and_fallback_prompts(self):
        engine = self.build_engine()
        minute_lines = [{"speaker": "Alice", "text": "Vamos iniciar"}]

        with patch.object(engine, "_request_json", return_value={}) as request_mock:
            engine.summarize_minute(minute_lines, "PROMPT_EXTERNO")
            engine.summarize_minute(minute_lines)

        self.assertEqual(request_mock.call_args_list[0].args[0], STT_APP.SUMMARY_KIND_MINUTE)
        self.assertEqual(request_mock.call_args_list[0].args[1], "minute-model")
        self.assertEqual(
            request_mock.call_args_list[0].args[2],
            "PROMPT_EXTERNO\n\n" + STT_APP.CONTRACT_SUFFIX_MINUTE,
        )
        self.assertEqual(
            request_mock.call_args_list[1].args[2],
            STT_APP.DEFAULT_SUMMARIZE_MINUTE_PROMPT.strip() + "\n\n" + STT_APP.CONTRACT_SUFFIX_MINUTE,
        )

    def test_merge_summaries_uses_external_and_fallback_prompts(self):
        engine = self.build_engine()
        previous_summary = STT_APP.default_accumulated_summary_payload()
        minute_summary = {
            "chunk_type": "mista",
            "topics": [],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }

        with patch.object(engine, "_request_json", return_value={}) as request_mock:
            engine.merge_summaries(previous_summary, minute_summary, "PROMPT_MERGE")
            engine.merge_summaries(previous_summary, minute_summary)

        self.assertEqual(request_mock.call_args_list[0].args[0], STT_APP.SUMMARY_KIND_ACCUMULATED)
        self.assertEqual(request_mock.call_args_list[0].args[1], "accumulated-model")
        self.assertEqual(
            request_mock.call_args_list[0].args[2],
            "PROMPT_MERGE\n\n" + STT_APP.CONTRACT_SUFFIX_ACCUMULATED,
        )
        self.assertEqual(
            request_mock.call_args_list[1].args[2],
            STT_APP.DEFAULT_MERGE_SUMMARIES_PROMPT.strip()
            + "\n\n"
            + STT_APP.CONTRACT_SUFFIX_ACCUMULATED,
        )

    def test_finalize_summary_uses_external_and_fallback_prompts(self):
        engine = self.build_engine()
        merged_summary = STT_APP.default_accumulated_summary_payload()

        with patch.object(engine, "_request_json", return_value={}) as request_mock:
            engine.finalize_summary(merged_summary, "PROMPT_FINAL")
            engine.finalize_summary(merged_summary)

        self.assertEqual(request_mock.call_args_list[0].args[0], STT_APP.SUMMARY_KIND_FINAL)
        self.assertEqual(request_mock.call_args_list[0].args[1], "final-model")
        self.assertEqual(
            request_mock.call_args_list[0].args[2],
            "PROMPT_FINAL\n\n" + STT_APP.CONTRACT_SUFFIX_FINAL,
        )
        self.assertEqual(
            request_mock.call_args_list[1].args[2],
            STT_APP.DEFAULT_FINALIZE_SUMMARY_PROMPT.strip() + "\n\n" + STT_APP.CONTRACT_SUFFIX_FINAL,
        )

    def test_finalize_summary_text_uses_external_and_fallback_prompts(self):
        engine = self.build_engine()
        final_summary = {
            "title": "Ata de Alinhamento",
            "conversation_types": [],
            "executive_summary": "Resumo.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }
        with patch.object(
            engine,
            "_request_text",
            return_value="# Ata de Reunião\n\nConteúdo final.",
        ) as request_mock:
            engine.finalize_summary_text(final_summary, "markdown", "PROMPT_ATA")
            engine.finalize_summary_text(final_summary, "markdown")

        self.assertEqual(request_mock.call_args_list[0].kwargs["kind"], None)
        self.assertEqual(request_mock.call_args_list[0].kwargs["model"], "final-text-model")
        self.assertIn("PROMPT_ATA", request_mock.call_args_list[0].kwargs["system_prompt"])
        self.assertIn(
            "Formato de saida: Markdown.",
            request_mock.call_args_list[0].kwargs["system_prompt"],
        )
        self.assertIn("Formato de saida esperado: markdown.", request_mock.call_args_list[0].kwargs["user_prompt"])
        self.assertIn(
            STT_APP.DEFAULT_FINALIZE_SUMMARY_TEXT_PROMPT.strip(),
            request_mock.call_args_list[1].kwargs["system_prompt"],
        )
        self.assertIn("Formato de saida: Markdown.", request_mock.call_args_list[1].kwargs["system_prompt"])

    def test_finalize_summary_text_accepts_valid_output_without_retry(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            return_value="# Ata de Reunião\n\nConteúdo final.",
        ) as request_mock:
            result = engine.finalize_summary_text(final_summary, "markdown")

        self.assertEqual(result, "# Ata de Reunião\n\nConteúdo final.")
        self.assertEqual(request_mock.call_count, 1)

    def test_finalize_summary_text_accepts_valid_html_without_retry(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            return_value="<h1>Ata de Reunião</h1>\n<section><p>Conteúdo final.</p></section>",
        ) as request_mock:
            result = engine.finalize_summary_text(final_summary, "html")

        self.assertTrue(result.startswith("<h1>Ata de Reunião</h1>"))
        self.assertEqual(request_mock.call_count, 1)
        self.assertIn("Formato de saida: HTML.", request_mock.call_args.kwargs["system_prompt"])

    def test_finalize_summary_text_retries_invalid_output_once(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            side_effect=[
                "We need to generate the ata first.\n# Ata de Reunião\n\nConteúdo.",
                "# Ata de Reunião\n\nConteúdo final.",
            ],
        ) as request_mock:
            result = engine.finalize_summary_text(final_summary, "markdown")

        self.assertEqual(result, "# Ata de Reunião\n\nConteúdo final.")
        self.assertEqual(request_mock.call_count, 2)
        self.assertIn(
            "A resposta anterior foi inválida",
            request_mock.call_args_list[1].kwargs["user_prompt"],
        )

    def test_finalize_summary_text_retries_html_markdown_then_accepts_html(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            side_effect=[
                "# Ata de Reunião\n\nConteúdo em markdown.",
                "<h1>Ata de Reunião</h1>\n<section><p>Conteúdo final.</p></section>",
            ],
        ) as request_mock:
            result = engine.finalize_summary_text(final_summary, "html")

        self.assertTrue(result.startswith("<h1>Ata de Reunião</h1>"))
        self.assertEqual(request_mock.call_count, 2)
        self.assertIn("HTML nao deve iniciar com Markdown", request_mock.call_args_list[1].kwargs["user_prompt"])

    def test_finalize_summary_text_raises_after_three_invalid_outputs(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            side_effect=[
                "We need to generate the ata first.",
                "The JSON contains a title.",
                "Still not an ata.",
            ],
        ) as request_mock:
            with self.assertRaisesRegex(RuntimeError, "ata textual final invalida"):
                engine.finalize_summary_text(final_summary, "markdown")

        self.assertEqual(request_mock.call_count, 3)

    def test_finalize_summary_text_raises_after_three_invalid_html_outputs(self):
        engine = self.build_engine()
        final_summary = {"title": "Ata de Alinhamento"}

        with patch.object(
            engine,
            "_request_text",
            side_effect=[
                "Antes da ata\n<h1>Ata de Reunião</h1>",
                "```html\n<h1>Ata de Reunião</h1>\n```",
                "# Ata de Reunião",
            ],
        ) as request_mock:
            with self.assertRaisesRegex(RuntimeError, "ata textual final invalida"):
                engine.finalize_summary_text(final_summary, "html")

        self.assertEqual(request_mock.call_count, 3)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummarySchemaValidationTests(unittest.TestCase):
    def test_parse_and_validate_summary_output_accepts_minute_payload(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "topics": [],
                "facts": [],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        parsed = STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)
        self.assertEqual(parsed["chunk_type"], "tecnica")

    def test_parse_and_validate_summary_output_rejects_invalid_payload(self):
        with self.assertRaises(RuntimeError):
            STT_APP.parse_and_validate_summary_output(
                STT_APP.SUMMARY_KIND_MINUTE,
                json.dumps({"chunk_type": "tecnica"}),
            )

    def test_parse_and_validate_summary_output_accepts_final_payload(self):
        raw = json.dumps(
            {
                "title": "Resumo Executivo: Cronograma de entrega",
                "conversation_types": ["executiva"],
                "executive_summary": "Resumo objetivo da reuniao.",
                "topics": [
                    {
                        "name": "Cronograma de entrega",
                        "summary": "Alinhado prazo inicial e dependencias.",
                        "decisions": ["Prazo preliminar confirmado para sexta."],
                        "pending_items": ["Confirmar disponibilidade da equipe de QA."],
                        "next_steps": ["Enviar plano consolidado ate amanha."],
                        "tags": ["cronograma"],
                    }
                ],
                "global_decisions": [
                    {"text": "Priorizar fase 1.", "confidence": "high", "tags": ["prioridade"]}
                ],
                "global_pending_items": [
                    {"text": "Definir responsavel tecnico.", "confidence": "medium", "tags": ["responsavel"]}
                ],
                "global_next_steps": [
                    {"text": "Agendar reuniao de acompanhamento.", "confidence": "medium", "tags": ["followup"]}
                ],
                "additional_notes": [
                    {"text": "Dependencia externa pode atrasar entrega.", "confidence": "low", "tags": ["risco"]}
                ],
            }
        )
        parsed = STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_FINAL, raw)
        self.assertEqual(parsed["title"], "Resumo Executivo: Cronograma de entrega")
        self.assertEqual(parsed["topics"][0]["name"], "Cronograma de entrega")

    def test_validate_minute_summary_payload_allows_more_than_six_topics(self):
        payload = {
            "chunk_type": "tecnica",
            "topics": [
                {
                    "name": f"Topico {index}",
                    "summary": f"Resumo {index}",
                    "status": "new",
                    "tags": [],
                }
                for index in range(7)
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        parsed = STT_APP.validate_minute_summary_payload(payload)
        self.assertEqual(len(parsed["topics"]), 7)

    def test_validate_accumulated_summary_payload_allows_more_than_twenty_topics(self):
        payload = {
            "conversation_types": [],
            "topics": [
                {
                    "name": f"Topico {index}",
                    "summary": f"Resumo {index}",
                    "status": "active",
                    "tags": [],
                    "facts": [],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [],
                }
                for index in range(21)
            ],
        }
        parsed = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(len(parsed["topics"]), 21)

    def test_validate_final_summary_payload_allows_more_than_twelve_topics(self):
        payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo final.",
            "topics": [
                {
                    "name": f"Topico {index}",
                    "summary": f"Resumo {index}",
                    "decisions": [],
                    "pending_items": [],
                    "next_steps": [],
                    "tags": [],
                }
                for index in range(13)
            ],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }
        parsed = STT_APP.validate_final_summary_payload(payload)
        self.assertEqual(len(parsed["topics"]), 13)

    def test_validate_final_summary_payload_truncates_global_lists_to_30(self):
        payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo final.",
            "topics": [],
            "global_decisions": [
                {"text": f"decision {i}", "confidence": "medium", "tags": []}
                for i in range(31)
            ],
            "global_pending_items": [
                {"text": f"pending {i}", "confidence": "medium", "tags": []}
                for i in range(31)
            ],
            "global_next_steps": [
                {"text": f"next {i}", "confidence": "medium", "tags": []}
                for i in range(31)
            ],
            "additional_notes": [
                {"text": f"note {i}", "confidence": "low", "tags": []}
                for i in range(31)
            ],
        }

        parsed = STT_APP.validate_final_summary_payload(payload)

        self.assertEqual(len(parsed["global_decisions"]), 30)
        self.assertEqual(len(parsed["global_pending_items"]), 30)
        self.assertEqual(len(parsed["global_next_steps"]), 30)
        self.assertEqual(len(parsed["additional_notes"]), 30)
        self.assertEqual(parsed["global_decisions"][0]["text"], "decision 0")
        self.assertEqual(parsed["global_decisions"][-1]["text"], "decision 29")

    def test_validate_final_summary_payload_truncates_topic_lists_to_30(self):
        payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo final.",
            "topics": [
                {
                    "name": "Cronograma",
                    "summary": "Resumo do cronograma.",
                    "decisions": [f"decision {i}" for i in range(31)],
                    "pending_items": [f"pending {i}" for i in range(31)],
                    "next_steps": [f"next {i}" for i in range(31)],
                    "tags": [],
                }
            ],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }

        parsed = STT_APP.validate_final_summary_payload(payload)
        topic = parsed["topics"][0]

        self.assertEqual(len(topic["decisions"]), 30)
        self.assertEqual(len(topic["pending_items"]), 30)
        self.assertEqual(len(topic["next_steps"]), 30)
        self.assertEqual(topic["decisions"][0], "decision 0")
        self.assertEqual(topic["decisions"][-1], "decision 29")

    def test_parse_and_validate_summary_output_rejects_legacy_final_fields(self):
        raw = json.dumps(
            {
                "title": "Resumo Final Executivo da Chamada",
                "conversation_types": [],
                "main_points": [],
                "decisions": [],
                "pending_items": [],
                "next_steps": [],
                "additional_notes": [],
            }
        )
        with self.assertRaises(RuntimeError):
            STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_FINAL, raw)

    def test_parse_and_validate_summary_output_rejects_missing_topics_in_minute(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "facts": [],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        with self.assertRaises(RuntimeError):
            STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)

    def test_parse_and_validate_summary_output_rejects_missing_topic_in_item(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "topics": [
                    {
                        "name": "Planejamento",
                        "summary": "Tema principal do trecho.",
                        "status": "new",
                        "tags": ["planejamento"],
                    }
                ],
                "facts": [
                    {
                        "text": "Equipe confirmou a agenda.",
                        "confidence": "high",
                        "status": "confirmed",
                        "tags": ["agenda"],
                    }
                ],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        with self.assertRaises(RuntimeError):
            STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)

    def test_parse_and_validate_summary_output_rejects_unknown_topic_reference(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "topics": [
                    {
                        "name": "Planejamento",
                        "summary": "Tema principal do trecho.",
                        "status": "new",
                        "tags": ["planejamento"],
                    }
                ],
                "facts": [
                    {
                        "text": "Equipe confirmou a agenda.",
                        "confidence": "high",
                        "status": "confirmed",
                        "tags": ["agenda"],
                        "topic": "Tema inexistente",
                    }
                ],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        with self.assertRaises(RuntimeError):
            STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)

    def test_parse_and_validate_summary_output_accepts_name_field_in_items(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "topics": [
                    {
                        "name": "Planejamento",
                        "summary": "Tema principal do trecho.",
                        "status": "new",
                        "tags": ["planejamento"],
                    }
                ],
                "facts": [
                    {
                        "text": "Equipe confirmou a agenda.",
                        "confidence": "high",
                        "status": "confirmed",
                        "tags": ["agenda"],
                        "name": "Planejamento",
                    }
                ],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        parsed = STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)
        self.assertEqual(parsed["facts"][0]["name"], "Planejamento")

    def test_parse_and_validate_summary_output_normalizes_topic_alias_to_name(self):
        raw = json.dumps(
            {
                "chunk_type": "tecnica",
                "topics": [
                    {
                        "name": "Planejamento",
                        "summary": "Tema principal do trecho.",
                        "status": "new",
                        "tags": ["planejamento"],
                    }
                ],
                "facts": [
                    {
                        "text": "Equipe confirmou a agenda.",
                        "confidence": "high",
                        "status": "confirmed",
                        "tags": ["agenda"],
                        "topic": "Planejamento",
                    }
                ],
                "hypotheses": [],
                "decisions": [],
                "open_items": [],
                "next_steps": [],
                "notes": [],
            }
        )
        parsed = STT_APP.parse_and_validate_summary_output(STT_APP.SUMMARY_KIND_MINUTE, raw)
        self.assertEqual(parsed["facts"][0]["name"], "Planejamento")

    def test_validate_minute_summary_payload_accepts_multiple_topics(self):
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Definicao de entregas da fase inicial.",
                    "status": "new",
                    "tags": ["escopo"],
                },
                {
                    "name": "Riscos operacionais",
                    "summary": "Levantados riscos de dependencia externa.",
                    "status": "continuing",
                    "tags": ["riscos"],
                },
            ],
            "facts": [
                {
                    "text": "A fase inicial inclui modulo de autenticacao.",
                    "confidence": "high",
                    "status": "confirmed",
                    "tags": ["escopo"],
                    "topic": "Escopo do projeto",
                }
            ],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [
                {
                    "text": "Dependencia externa ainda sem prazo final.",
                    "confidence": "low",
                    "status": "uncertain",
                    "tags": ["riscos"],
                    "topic": "Riscos operacionais",
                }
            ],
        }
        normalized = STT_APP.validate_minute_summary_payload(payload)
        self.assertEqual(len(normalized["topics"]), 2)

    def test_validate_minute_summary_payload_truncates_sections_to_20(self):
        topic_name = "Escopo do projeto"
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": topic_name,
                    "summary": "Tema principal.",
                    "status": "new",
                    "tags": ["escopo"],
                }
            ],
            "facts": [
                {
                    "text": f"fact {i}",
                    "confidence": "high",
                    "status": "confirmed",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
            "hypotheses": [
                {
                    "text": f"hypothesis {i}",
                    "confidence": "low",
                    "status": "uncertain",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
            "decisions": [
                {
                    "text": f"decision {i}",
                    "confidence": "high",
                    "status": "confirmed",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
            "open_items": [
                {
                    "text": f"open {i}",
                    "confidence": "medium",
                    "status": "open",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
            "next_steps": [
                {
                    "text": f"next {i}",
                    "confidence": "medium",
                    "status": "planned",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
            "notes": [
                {
                    "text": f"note {i}",
                    "confidence": "low",
                    "status": "info",
                    "tags": ["escopo"],
                    "topic": topic_name,
                }
                for i in range(21)
            ],
        }

        normalized = STT_APP.validate_minute_summary_payload(payload)

        self.assertEqual(len(normalized["facts"]), 20)
        self.assertEqual(len(normalized["hypotheses"]), 20)
        self.assertEqual(len(normalized["decisions"]), 20)
        self.assertEqual(len(normalized["open_items"]), 20)
        self.assertEqual(len(normalized["next_steps"]), 20)
        self.assertEqual(len(normalized["notes"]), 20)
        self.assertEqual(normalized["facts"][0]["text"], "fact 0")
        self.assertEqual(normalized["facts"][-1]["text"], "fact 19")

    def test_validate_accumulated_summary_payload_accepts_items_inside_topics(self):
        payload = {
            "conversation_types": ["mista"],
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Escopo consolidado da fase inicial.",
                    "status": "active",
                    "tags": ["escopo"],
                    "facts": [
                        {
                            "text": "Escopo da fase 1 foi confirmado.",
                            "confidence": "high",
                            "status": "confirmed",
                            "tags": ["escopo"],
                        }
                    ],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [],
                }
            ],
        }
        normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(normalized["topics"][0]["status"], "active")
        self.assertEqual(normalized["topics"][0]["facts"][0]["text"], "Escopo da fase 1 foi confirmado.")

    def test_validate_accumulated_summary_payload_accepts_all_topic_sections(self):
        payload = {
            "conversation_types": ["mista"],
            "topics": [
                {
                    "name": "Integração com LiveKit",
                    "summary": "Resumo consolidado do assunto.",
                    "status": "active",
                    "tags": ["livekit"],
                    "facts": [
                        {"text": "Fato confirmado.", "confidence": "high", "status": "confirmed", "tags": []}
                    ],
                    "hypotheses": [
                        {"text": "Hipotese em aberto.", "confidence": "medium", "status": "uncertain", "tags": []}
                    ],
                    "decisions": [
                        {"text": "Decisao registrada.", "confidence": "high", "status": "confirmed", "tags": []}
                    ],
                    "open_items": [
                        {"text": "Pendencia registrada.", "confidence": "medium", "status": "open", "tags": []}
                    ],
                    "next_steps": [
                        {"text": "Proximo passo planejado.", "confidence": "medium", "status": "planned", "tags": []}
                    ],
                    "notes": [
                        {"text": "Observacao adicional.", "confidence": "low", "status": "info", "tags": []}
                    ],
                }
            ],
        }

        normalized = STT_APP.validate_accumulated_summary_payload(payload)
        topic = normalized["topics"][0]

        self.assertEqual(topic["name"], "Integração com LiveKit")
        self.assertEqual(topic["decisions"][0]["text"], "Decisao registrada.")
        self.assertEqual(topic["open_items"][0]["status"], "open")

    def test_validate_accumulated_summary_payload_rejects_name_inside_topic_item(self):
        payload = {
            "conversation_types": ["mista"],
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Escopo consolidado da fase inicial.",
                    "status": "active",
                    "tags": ["escopo"],
                    "facts": [
                        {
                            "text": "Aprovado seguir com homologacao.",
                            "confidence": "high",
                            "status": "confirmed",
                            "tags": ["homologacao"],
                            "name": "Homologacao",
                        }
                    ],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [],
                }
            ],
        }
        with self.assertRaises(RuntimeError):
            STT_APP.validate_accumulated_summary_payload(payload)

    def test_validate_accumulated_summary_payload_rejects_high_confidence_notes_and_hypotheses(self):
        payload = {
            "conversation_types": ["mista"],
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Escopo consolidado da fase inicial.",
                    "status": "active",
                    "tags": ["escopo"],
                    "facts": [],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [
                        {
                            "text": "Nota nao deve ter confianca alta.",
                            "confidence": "high",
                            "status": "info",
                            "tags": [],
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(RuntimeError):
            STT_APP.validate_accumulated_summary_payload(payload)

        payload["topics"][0]["notes"] = []
        payload["topics"][0]["hypotheses"] = [
            {
                "text": "Hipotese nao deve ter confianca alta.",
                "confidence": "high",
                "status": "uncertain",
                "tags": [],
            }
        ]
        with self.assertRaises(RuntimeError):
            STT_APP.validate_accumulated_summary_payload(payload)

    def test_validate_minute_summary_payload_rejects_invalid_topic_status(self):
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Tema principal.",
                    "status": "active",
                    "tags": ["escopo"],
                }
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        with self.assertRaises(RuntimeError):
            STT_APP.validate_minute_summary_payload(payload)

    def test_validate_minute_summary_payload_rejects_invalid_notes_status(self):
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Tema principal.",
                    "status": "new",
                    "tags": ["escopo"],
                }
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [
                {
                    "text": "Nota invalida",
                    "confidence": "low",
                    "status": "open",
                    "tags": ["escopo"],
                    "topic": "Escopo do projeto",
                }
            ],
        }
        with self.assertRaises(RuntimeError):
            STT_APP.validate_minute_summary_payload(payload)

    def test_validate_summary_tags_allows_empty_and_missing(self):
        self.assertEqual(STT_APP.validate_summary_tags([], "item"), [])
        self.assertEqual(STT_APP.validate_summary_tags(None, "item"), [])

    def test_validate_summary_tags_truncates_above_4(self):
        tags = ["a", "b", "c", "d", "e", "f"]
        normalized = STT_APP.validate_summary_tags(tags, "item")
        self.assertEqual(normalized, ["a", "b", "c", "d"])


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class AccumulatedSummaryLimitsTests(unittest.TestCase):
    def _item(self, text: str, status: str, confidence: str = "medium") -> dict:
        return {
            "text": text,
            "confidence": confidence,
            "status": status,
            "tags": ["tag"],
        }

    def _payload(self, notes_count: int = 0, facts_count: int = 0) -> dict:
        return {
            "conversation_types": [],
            "topics": [
                {
                    "name": "topico_1",
                    "summary": "topico consolidado",
                    "status": "active",
                    "tags": ["tag"],
                    "facts": [self._item(f"fact {i}", "confirmed", "high") for i in range(facts_count)],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [self._item(f"note {i}", "info") for i in range(notes_count)],
                }
            ],
        }

    def test_validate_accumulated_summary_accepts_40_notes(self):
        with patch.object(STT_APP, "SUMMARY_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(notes_count=40)
            normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(len(normalized["topics"][0]["notes"]), 40)

    def test_validate_accumulated_summary_truncates_41_notes_to_40(self):
        with patch.object(STT_APP, "SUMMARY_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(notes_count=41)
            normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(len(normalized["topics"][0]["notes"]), 40)
        self.assertEqual(normalized["topics"][0]["notes"][0]["text"], "note 0")
        self.assertEqual(normalized["topics"][0]["notes"][-1]["text"], "note 39")

    def test_validate_accumulated_summary_truncates_41_facts_to_40(self):
        with patch.object(STT_APP, "SUMMARY_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(facts_count=41)
            normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(len(normalized["topics"][0]["facts"]), 40)
        self.assertEqual(normalized["topics"][0]["facts"][0]["text"], "fact 0")
        self.assertEqual(normalized["topics"][0]["facts"][-1]["text"], "fact 39")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryEngineRetryTests(unittest.TestCase):
    def build_engine(self) -> object:
        return STT_APP.SummaryEngine(
            enabled=True,
            provider="openai",
            api_key="test-key",
            base_url="https://api.openai.com/v1",
            extra_headers={},
            minute_model="minute-model",
            accumulated_model="accumulated-model",
            final_model="final-model",
            final_text_model="final-text-model",
            timeout_seconds=45,
            request_retries=2,
            retry_base_seconds=1.5,
        )

    def _response(self, body: dict):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(body).encode("utf-8")

        return _Response()

    def _minute_payload(self) -> dict:
        return {
            "chunk_type": "mista",
            "topics": [],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }

    def _accumulated_payload(self) -> dict:
        return {
            "conversation_types": [],
            "topics": [],
        }

    def _final_payload(self) -> dict:
        return {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo executivo objetivo.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }

    def test_request_text_retries_and_succeeds_after_timeouts(self):
        engine = self.build_engine()
        success_payload = {"output_text": "{}"}
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("timeout-1"), TimeoutError("timeout-2"), self._response(success_payload)],
        ) as urlopen_mock:
            with patch.object(STT_APP.time, "sleep") as sleep_mock:
                raw = engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "openai/gpt-5.6-luna", "system", "user")

        self.assertEqual(raw, "{}")
        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_request_text_raises_after_exhausting_retries(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("timeout-1"), TimeoutError("timeout-2"), TimeoutError("timeout-3")],
        ) as urlopen_mock:
            with patch.object(STT_APP.time, "sleep") as sleep_mock:
                with self.assertRaises(RuntimeError):
                    engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "openai/gpt-5.6-luna", "system", "user")

        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)

    def test_request_text_sends_strict_json_schema_format_for_minute(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "{}"}),
        ) as urlopen_mock:
            raw = engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")

        self.assertEqual(raw, "{}")
        req = urlopen_mock.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertEqual(body["text"]["format"]["name"], "summary_minute_v1")
        self.assertEqual(
            body["text"]["format"]["schema"]["properties"]["chunk_type"]["enum"],
            ["tecnica", "executiva", "operacional", "comercial", "mista"],
        )
        self.assertEqual(req.full_url, "https://api.openai.com/v1/responses")

    def test_request_text_sends_strict_json_schema_format_for_accumulated_and_final(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "{}"}),
        ) as urlopen_mock:
            engine._request_text(STT_APP.SUMMARY_KIND_ACCUMULATED, "accumulated-model", "system", "user")
        req_acc = urlopen_mock.call_args.args[0]
        body_acc = json.loads(req_acc.data.decode("utf-8"))
        self.assertEqual(body_acc["text"]["format"]["name"], "summary_accumulated_v1")
        self.assertEqual(
            body_acc["text"]["format"]["schema"]["properties"]["topics"]["items"]["properties"]["status"]["enum"],
            ["active", "open", "resolved", "uncertain"],
        )

        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "{}"}),
        ) as urlopen_mock:
            engine._request_text(STT_APP.SUMMARY_KIND_FINAL, "final-model", "system", "user")
        req_final = urlopen_mock.call_args.args[0]
        body_final = json.loads(req_final.data.decode("utf-8"))
        self.assertEqual(body_final["text"]["format"]["name"], "summary_final_v1")
        self.assertIn("executive_summary", body_final["text"]["format"]["schema"]["required"])
        title_schema = body_final["text"]["format"]["schema"]["properties"]["title"]
        self.assertEqual(title_schema["type"], "string")
        self.assertEqual(title_schema["minLength"], STT_APP.SUMMARY_FINAL_TITLE_MIN_CHARS)
        self.assertEqual(title_schema["maxLength"], STT_APP.SUMMARY_FINAL_TITLE_MAX_CHARS)
        self.assertNotIn("enum", title_schema)

    def test_request_text_without_kind_does_not_send_json_schema(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "ata final"}),
        ) as urlopen_mock:
            raw = engine._request_text(None, "final-text-model", "system", "user")

        self.assertEqual(raw, "ata final")
        req = urlopen_mock.call_args.args[0]
        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("text", body)

    def test_request_text_includes_openrouter_optional_headers(self):
        engine = STT_APP.SummaryEngine(
            enabled=True,
            provider="openrouter",
            api_key="router-key",
            base_url="https://openrouter.ai/api/v1",
            extra_headers={"HTTP-Referer": "https://app.local", "X-Title": "Talk STT"},
            minute_model="minute-model",
            accumulated_model="accumulated-model",
            final_model="final-model",
            final_text_model="final-text-model",
            timeout_seconds=45,
            request_retries=2,
            retry_base_seconds=1.5,
        )
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "{}"}),
        ) as urlopen_mock:
            engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")
        req = urlopen_mock.call_args.args[0]
        headers = {str(k).lower(): str(v) for k, v in req.headers.items()}
        self.assertEqual(req.full_url, "https://openrouter.ai/api/v1/responses")
        self.assertEqual(headers.get("http-referer"), "https://app.local")
        self.assertEqual(headers.get("x-title"), "Talk STT")

    def test_request_text_raises_on_model_refusal(self):
        engine = self.build_engine()
        refusal_payload = {
            "output": [
                {
                    "content": [
                        {
                            "type": "refusal",
                            "text": "nao posso atender este pedido",
                        }
                    ]
                }
            ]
        }
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response(refusal_payload),
        ):
            with self.assertRaises(RuntimeError):
                engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")

    def test_request_json_succeeds_for_each_kind_with_valid_payload(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": json.dumps(self._minute_payload())}),
        ):
            minute = engine._request_json(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")
        self.assertEqual(minute["chunk_type"], "mista")

        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": json.dumps(self._accumulated_payload())}),
        ):
            accumulated = engine._request_json(
                STT_APP.SUMMARY_KIND_ACCUMULATED, "accumulated-model", "system", "user"
            )
        self.assertEqual(accumulated["conversation_types"], [])

        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": json.dumps(self._final_payload())}),
        ):
            final = engine._request_json(STT_APP.SUMMARY_KIND_FINAL, "final-model", "system", "user")
        self.assertEqual(final["title"], "Resumo Final Executivo da Chamada")

    def test_request_json_fail_closed_when_model_returns_non_json(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": "nao-json"}),
        ):
            with self.assertRaises(STT_APP.SummaryContractValidationError):
                engine._request_json(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")

    def test_request_text_fail_closed_when_model_returns_empty_output(self):
        engine = self.build_engine()
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output": []}),
        ):
            with self.assertRaises(RuntimeError):
                engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")

    def test_request_json_raises_typed_contract_error_with_raw_output(self):
        engine = self.build_engine()
        raw_output = json.dumps({"chunk_type": "tecnica"})
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            return_value=self._response({"output_text": raw_output}),
        ):
            with self.assertRaises(STT_APP.SummaryContractValidationError) as ctx:
                engine._request_json(STT_APP.SUMMARY_KIND_MINUTE, "minute-model", "system", "user")
        self.assertEqual(ctx.exception.kind, STT_APP.SUMMARY_KIND_MINUTE)
        self.assertEqual(ctx.exception.model, "minute-model")
        self.assertEqual(ctx.exception.raw_output, raw_output)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class RoutingPathTests(unittest.TestCase):
    def test_build_paths_matches_expected_structure(self):
        doc_path = STT_APP.build_firestore_doc_path(
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            call_session_id="RM_session-1",
        )
        storage_base = STT_APP.build_storage_base_path(
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            call_session_id="RM_session-1",
        )
        self.assertEqual(
            doc_path,
            "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
        )
        self.assertEqual(
            storage_base,
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class MinuteWindowTests(unittest.TestCase):
    def test_compute_minute_index_uses_session_relative_window(self):
        session_started_at = "2026-04-18T10:00:00+00:00"
        chunk_ended_at = "2026-04-18T10:01:10+00:00"
        minute_index = STT_APP.compute_minute_index(
            session_started_at,
            chunk_ended_at,
            60,
        )
        self.assertEqual(minute_index, 1)

    def test_compute_minute_index_with_custom_window(self):
        session_started_at = "2026-04-18T10:00:00+00:00"
        chunk_ended_at = "2026-04-18T10:01:10+00:00"
        minute_index = STT_APP.compute_minute_index(
            session_started_at,
            chunk_ended_at,
            30,
        )
        self.assertEqual(minute_index, 2)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryProviderConfigTests(unittest.TestCase):
    def test_load_summary_api_key_returns_key_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "openai_apikey.json"
            secret_path.write_text(json.dumps({"api_key": "test-key"}), encoding="utf-8")
            with patch.object(STT_APP, "SUMMARY_ENABLED", True):
                with patch.object(STT_APP, "OPENAI_APIKEY_FILE", secret_path):
                    self.assertEqual(STT_APP.load_summary_api_key("openai"), "test-key")

    def test_load_summary_api_key_uses_openrouter_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "openrouter_apikey.json"
            secret_path.write_text(json.dumps({"api_key": "router-key"}), encoding="utf-8")
            with patch.object(STT_APP, "SUMMARY_ENABLED", True):
                with patch.object(STT_APP, "OPENROUTER_APIKEY_FILE", secret_path):
                    self.assertEqual(STT_APP.load_summary_api_key("openrouter"), "router-key")

    def test_summary_provider_defaults_to_openrouter_when_env_missing(self):
        self.assertEqual(STT_APP.SUMMARY_PROVIDER, "openrouter")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class ConfigDefaultsTests(unittest.TestCase):
    def test_openai_request_timeout_default_is_300_in_source(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("SUMMARY_REQUEST_TIMEOUT_SECONDS", "300")', source)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class CallIndexContractTests(unittest.TestCase):
    def build_routing(self) -> object:
        return STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path=(
                "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1"
            ),
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )

    def test_publish_call_index_writes_new_ready_fields_when_explicit(self):
        fake_doc = MagicMock()
        fake_fs_client = MagicMock()
        fake_fs_client.document.return_value = fake_doc
        sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        sink.publish_call_index(
            routing=self.build_routing(),
            status="finalized",
            last_minute_index=8,
            finalized=True,
            minute_window_seconds=60,
            flush_interval_seconds=30,
            final_summary_path="path/final_summary.json",
            summary_accumulated_path="path/accumulated.json",
            final_summary_ready=True,
            final_transcript_path="path/final_transcript.json",
            final_transcript_ready=True,
            final_summary_text_path="path/final_summary_text.txt",
            final_summary_text_ready=True,
        )

        payload = fake_doc.set.call_args.args[0]
        self.assertEqual(payload["status"], "finalized")
        self.assertTrue(payload["finalized"])
        self.assertEqual(payload["final_summary_path"], "path/final_summary.json")
        self.assertTrue(payload["final_summary_ready"])
        self.assertEqual(payload["final_transcript_path"], "path/final_transcript.json")
        self.assertTrue(payload["final_transcript_ready"])
        self.assertEqual(payload["final_summary_text_path"], "path/final_summary_text.txt")
        self.assertTrue(payload["final_summary_text_ready"])

    def test_publish_call_index_skips_optional_fields_when_unset(self):
        fake_doc = MagicMock()
        fake_fs_client = MagicMock()
        fake_fs_client.document.return_value = fake_doc
        sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        sink.publish_call_index(
            routing=self.build_routing(),
            status="processing",
            last_minute_index=2,
            finalized=False,
            minute_window_seconds=60,
            flush_interval_seconds=30,
        )

        payload = fake_doc.set.call_args.args[0]
        self.assertNotIn("final_summary_path", payload)
        self.assertNotIn("final_summary_ready", payload)
        self.assertNotIn("final_transcript_path", payload)
        self.assertNotIn("final_transcript_ready", payload)
        self.assertNotIn("final_summary_text_path", payload)
        self.assertNotIn("final_summary_text_ready", payload)

    def test_upsert_room_session_links_on_start_writes_only_session_doc(self):
        session_ref = MagicMock()
        fake_fs_client = MagicMock()
        fake_fs_client.document.return_value = session_ref
        sink = STT_APP.FirebaseSink(
            namespace="talk__dev",
            enabled=True,
            firestore_client=fake_fs_client,
            storage_bucket=None,
        )

        sink.upsert_room_session_links_on_start(self.build_routing())

        fake_fs_client.document.assert_called_once_with(
            "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1"
        )
        payload = session_ref.set.call_args.args[0]
        self.assertEqual(payload["call_session_id"], "RM_session-1")
        self.assertEqual(payload["transcript_session_id"], "session-1")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryToleranceAndFinalizationClaimTests(unittest.TestCase):
    def test_validate_minute_summary_payload_normalizes_invalid_notes_confidence(self):
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "ruido",
                    "summary": "Trecho com informacoes pouco confiaveis.",
                    "status": "new",
                    "tags": ["ruido"],
                }
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [
                {
                    "text": "ponto com baixa confianca",
                    "confidence": "HIGH",
                    "status": "info",
                    "tags": ["ruido"],
                    "topic": "ruido",
                }
            ],
        }
        normalized = STT_APP.validate_minute_summary_payload(payload)
        self.assertEqual(normalized["notes"][0]["confidence"], "medium")

    def test_validate_minute_summary_payload_normalizes_invalid_hypotheses_confidence(self):
        payload = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "hipoteses",
                    "summary": "Pontos ainda nao confirmados.",
                    "status": "new",
                    "tags": ["hipotese"],
                }
            ],
            "facts": [],
            "hypotheses": [
                {
                    "text": "hipotese inicial",
                    "confidence": "unknown",
                    "status": "uncertain",
                    "tags": ["hipotese"],
                    "topic": "hipoteses",
                }
            ],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        normalized = STT_APP.validate_minute_summary_payload(payload)
        self.assertEqual(normalized["hypotheses"][0]["confidence"], "medium")

    def test_db_claim_room_finalization_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    now = "2026-04-22T16:30:00+00:00"
                    conn = STT_APP.read_db_connection()
                    try:
                        conn.execute(
                            """
                            INSERT INTO sessions(
                                room_name, session_id, call_session_id, transcript_session_id, room_id, vertical, slug,
                                firestore_doc_path, storage_base_path,
                                started_at, state, room_end_received, last_chunk_at, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'room_ended', 1, ?, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                "RM_session-1",
                                None,
                                "roomA",
                                "HEALTH",
                                "acme",
                                "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
                                "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
                                now,
                                now,
                                now,
                                now,
                            ),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    first = STT_APP.db_claim_room_finalization(
                        "talk__dev__roomA", "RM_session-1", now
                    )
                    second = STT_APP.db_claim_room_finalization(
                        "talk__dev__roomA", "RM_session-1", now
                    )

                    self.assertTrue(first)
                    self.assertFalse(second)

                    row = STT_APP.db_get_session_row("talk__dev__roomA", "RM_session-1")
                    self.assertEqual(row["state"], "finalizing")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryFinalResilienceTests(unittest.TestCase):
    def test_build_accumulated_from_minute_summaries_generates_valid_payload(self):
        minute = {
            "chunk_type": "mista",
            "topics": [
                {
                    "name": "Planejamento",
                    "summary": "Escopo inicial discutido.",
                    "status": "new",
                    "tags": ["escopo"],
                }
            ],
            "facts": [
                {
                    "text": "Prazo preliminar definido para sexta.",
                    "confidence": "high",
                    "status": "confirmed",
                    "tags": ["prazo"],
                    "name": "Planejamento",
                }
            ],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        accumulated = STT_APP.build_accumulated_from_minute_summaries([minute])
        self.assertEqual(accumulated["conversation_types"], ["mista"])
        self.assertEqual(accumulated["topics"][0]["status"], "active")
        self.assertEqual(accumulated["topics"][0]["facts"][0]["text"], "Prazo preliminar definido para sexta.")
        self.assertNotIn("name", accumulated["topics"][0]["facts"][0])

    def test_build_deterministic_final_summary_includes_partial_disclosure(self):
        accumulated = {
            "conversation_types": ["executiva"],
            "topics": [
                {
                    "name": "Cronograma",
                    "summary": "Discussao do cronograma.",
                    "status": "active",
                    "tags": ["prazo"],
                    "facts": [],
                    "hypotheses": [],
                    "decisions": [
                        {
                            "text": "Follow-up marcado para sexta.",
                            "confidence": "high",
                            "status": "confirmed",
                            "tags": ["followup"],
                        }
                    ],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [],
                }
            ],
        }
        final_payload = STT_APP.build_deterministic_final_summary(
            accumulated,
            [2, 5],
            ["acumulado reconstruido"],
        )
        self.assertTrue(final_payload["title"].startswith("Documento Parcial:"))
        self.assertIn("Cronograma", final_payload["title"])
        self.assertIn("Documento parcial", final_payload["executive_summary"])
        self.assertTrue(
            any("Ata parcial" in note["text"] for note in final_payload["additional_notes"])
        )
        self.assertEqual(final_payload["topics"][0]["decisions"], ["Follow-up marcado para sexta."])
        self.assertEqual(final_payload["global_decisions"][0]["text"], "Follow-up marcado para sexta.")

    def test_inject_degradation_disclosure_adds_summary_and_note(self):
        final_payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": ["executiva"],
            "executive_summary": "Resumo principal da chamada.",
            "topics": [
                {
                    "name": "Cronograma",
                    "summary": "Resumo do tema.",
                    "decisions": [],
                    "pending_items": [],
                    "next_steps": [],
                    "tags": [],
                }
            ],
            "global_decisions": [
                {"text": "Decisao A", "confidence": "medium", "tags": []}
            ],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }
        updated = STT_APP.inject_degradation_disclosure(
            final_payload,
            [1, 4],
            ["minutos ausentes"],
        )
        self.assertIn("Documento parcial", updated["executive_summary"])
        self.assertEqual(len(updated["additional_notes"]), 1)
        self.assertIn("Ata parcial", updated["additional_notes"][0]["text"])

    def test_inject_degradation_disclosure_truncates_additional_notes_to_30(self):
        final_payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": ["executiva"],
            "executive_summary": "Resumo principal da chamada.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [
                {"text": f"nota {i}", "confidence": "low", "tags": []}
                for i in range(35)
            ],
        }
        updated = STT_APP.inject_degradation_disclosure(
            final_payload,
            [3],
            ["minutos ausentes"],
        )
        self.assertEqual(len(updated["additional_notes"]), 30)
        self.assertEqual(updated["additional_notes"][0]["text"], "nota 0")
        self.assertEqual(updated["additional_notes"][-1]["text"], "nota 29")

    def test_build_deterministic_final_summary_truncates_notes_to_30(self):
        accumulated = {
            "conversation_types": ["executiva"],
            "topics": [
                {
                    "name": "Cronograma",
                    "summary": "Discussao do cronograma.",
                    "status": "active",
                    "tags": ["prazo"],
                    "facts": [],
                    "hypotheses": [],
                    "decisions": [],
                    "open_items": [],
                    "next_steps": [],
                    "notes": [
                        {
                            "text": f"note {i}",
                            "confidence": "low",
                            "status": "info",
                            "tags": [],
                        }
                        for i in range(40)
                    ],
                }
            ],
        }
        final_payload = STT_APP.build_deterministic_final_summary(
            accumulated,
            [2],
            ["acumulado reconstruido"],
        )
        self.assertEqual(len(final_payload["additional_notes"]), 30)
        self.assertEqual(final_payload["additional_notes"][0]["text"], "note 0")
        self.assertEqual(final_payload["additional_notes"][-1]["text"], "note 29")

    def test_db_schedule_final_summary_task_force_requeues_done(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    now = "2026-04-23T15:00:00+00:00"
                    conn = STT_APP.read_db_connection()
                    try:
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, -1, 'done', 2, ?, NULL, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", now, now, now),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    changed = STT_APP.db_schedule_final_summary_task(
                        "talk__dev__roomA",
                        "RM_session-1",
                        now,
                        False,
                    )
                    self.assertFalse(changed)

                    changed_force = STT_APP.db_schedule_final_summary_task(
                        "talk__dev__roomA",
                        "RM_session-1",
                        now,
                        True,
                    )
                    self.assertTrue(changed_force)
                    rows = STT_APP.db_get_summary_task_rows("talk__dev__roomA", "RM_session-1")
                    final_row = [row for row in rows if int(row["minute_index"]) < 0][0]
                    self.assertEqual(final_row["status"], "pending")
                    self.assertEqual(int(final_row["retries"]), 0)

    def test_db_recover_stale_summary_tasks_moves_processing_to_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    updated_at = "2026-04-23T14:00:00+00:00"
                    now = "2026-04-23T14:10:00+00:00"
                    conn = STT_APP.read_db_connection()
                    try:
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, ?, 'processing', 0, ?, NULL, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                2,
                                updated_at,
                                updated_at,
                                updated_at,
                            ),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    recovered = STT_APP.db_recover_stale_summary_tasks(now, 300)
                    self.assertEqual(recovered, 1)
                    rows = STT_APP.db_get_summary_task_rows("talk__dev__roomA", "RM_session-1")
                    self.assertEqual(rows[0]["status"], "error")
                    self.assertEqual(int(rows[0]["retries"]), 1)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryWorkerFinalContractErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_final_contract_error_uploads_temp_and_keeps_final_not_ready(self):
        stop_event = asyncio.Event()
        routing = STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path="VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )

        router = MagicMock()
        router.fetch_json.return_value = {"summary": STT_APP.default_accumulated_summary_payload()}
        router.fetch_agent_prompt.return_value = None
        router.upload_json = MagicMock()
        router.publish_call_index = MagicMock()

        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.final_model = "final-model"
        summary_engine.finalize_summary.side_effect = STT_APP.SummaryContractValidationError(
            kind=STT_APP.SUMMARY_KIND_FINAL,
            model="final-model",
            raw_output='{"topics":[{"name":"A"}]}',
            message="final.topics[0].summary deve ser string",
        )

        task = {
            "room_name": "talk__dev__roomA",
            "call_session_id": "RM_session-1",
            "minute_index": -1,
            "retries": 0,
        }
        session_row = {"finalized_at": "2026-04-24T12:00:00+00:00"}
        final_task_rows = [
            {
                "minute_index": -1,
                "status": "processing",
                "retries": 0,
                "updated_at": "2026-04-24T12:00:00+00:00",
            }
        ]

        claim_calls = {"count": 0}

        def fake_claim_summary_task(_now_iso: str):
            if claim_calls["count"] == 0:
                claim_calls["count"] += 1
                return task
            stop_event.set()
            return None

        done_calls: list[tuple[str, str, int]] = []
        error_calls: list[tuple[str, str, int, int, str]] = []

        def fake_mark_done(room_name: str, call_session_id: str, minute_index: int, _now_iso: str):
            done_calls.append((room_name, call_session_id, minute_index))
            stop_event.set()

        def fake_mark_error(
            room_name: str,
            call_session_id: str,
            minute_index: int,
            retries: int,
            error_message: str,
            _now_iso: str,
        ):
            error_calls.append((room_name, call_session_id, minute_index, retries, error_message))
            stop_event.set()

        with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
            with patch.object(STT_APP, "db_claim_summary_task", side_effect=fake_claim_summary_task):
                with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
                    with patch.object(STT_APP, "routing_context_from_session_row", return_value=routing):
                        with patch.object(STT_APP, "db_get_summary_task_rows", return_value=final_task_rows):
                            with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                                with patch.object(STT_APP, "db_mark_summary_task_done", side_effect=fake_mark_done):
                                    with patch.object(
                                        STT_APP,
                                        "db_mark_summary_task_error",
                                        side_effect=fake_mark_error,
                                    ):
                                        await asyncio.wait_for(
                                            STT_APP.summary_worker_loop(stop_event, router, summary_engine),
                                            timeout=2,
                                        )

        self.assertEqual(done_calls, [])
        self.assertEqual(len(error_calls), 1)
        self.assertEqual(error_calls[0][:4], ("talk__dev__roomA", "RM_session-1", -1, 1))
        self.assertIn("falha de contrato", error_calls[0][4])

        upload_paths = [call.args[1] for call in router.upload_json.call_args_list]
        self.assertIn(
            STT_APP.session_final_summary_temp_path(routing.storage_base_path),
            upload_paths,
        )
        self.assertNotIn(
            STT_APP.session_final_summary_path(routing.storage_base_path),
            upload_paths,
        )

        temp_upload_call = next(
            call
            for call in router.upload_json.call_args_list
            if call.args[1] == STT_APP.session_final_summary_temp_path(routing.storage_base_path)
        )
        temp_payload = temp_upload_call.args[2]
        self.assertEqual(temp_payload["model"], "final-model")
        self.assertEqual(temp_payload["kind"], STT_APP.SUMMARY_KIND_FINAL)
        self.assertIn('"name":"A"', temp_payload["raw_output"])

        publish_kwargs = router.publish_call_index.call_args.kwargs
        self.assertFalse(publish_kwargs["final_summary_ready"])
        self.assertFalse(publish_kwargs["final_summary_text_ready"])

    async def test_final_summary_generates_text_and_marks_ready(self):
        stop_event = asyncio.Event()
        routing = STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path="VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )
        final_payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo executivo objetivo.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }

        router = MagicMock()
        router.fetch_json.return_value = {"summary": STT_APP.default_accumulated_summary_payload()}
        router.fetch_agent_prompt.return_value = None
        router.upload_json = MagicMock()
        router.upload_text = MagicMock()
        router.publish_call_index = MagicMock()

        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.final_model = "final-model"
        summary_engine.final_text_model = "final-text-model"
        summary_engine.finalize_summary.return_value = final_payload
        summary_engine.finalize_summary_text.return_value = "# Ata Final"

        task = {
            "room_name": "talk__dev__roomA",
            "call_session_id": "RM_session-1",
            "minute_index": -1,
            "retries": 0,
        }
        session_row = {"finalized_at": "2026-04-24T12:00:00+00:00"}
        final_task_rows = [
            {
                "minute_index": -1,
                "status": "processing",
                "retries": 0,
                "updated_at": "2026-04-24T12:00:00+00:00",
            }
        ]

        claim_calls = {"count": 0}

        def fake_claim_summary_task(_now_iso: str):
            if claim_calls["count"] == 0:
                claim_calls["count"] += 1
                return task
            stop_event.set()
            return None

        done_calls: list[tuple[str, str, int]] = []

        def fake_mark_done(room_name: str, call_session_id: str, minute_index: int, _now_iso: str):
            done_calls.append((room_name, call_session_id, minute_index))
            stop_event.set()

        with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
            with patch.object(STT_APP, "db_claim_summary_task", side_effect=fake_claim_summary_task):
                with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
                    with patch.object(STT_APP, "routing_context_from_session_row", return_value=routing):
                        with patch.object(STT_APP, "db_get_summary_task_rows", return_value=final_task_rows):
                            with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                                with patch.object(STT_APP, "db_mark_summary_task_done", side_effect=fake_mark_done):
                                    with patch.object(STT_APP, "db_mark_summary_task_error") as mark_error_mock:
                                        await asyncio.wait_for(
                                            STT_APP.summary_worker_loop(stop_event, router, summary_engine),
                                            timeout=2,
                                        )

        self.assertEqual(done_calls, [("talk__dev__roomA", "RM_session-1", -1)])
        mark_error_mock.assert_not_called()
        router.upload_text.assert_called_once()
        upload_text_args = router.upload_text.call_args.args
        self.assertEqual(
            upload_text_args[1],
            STT_APP.session_final_summary_text_path(routing.storage_base_path),
        )
        self.assertEqual(upload_text_args[2], "# Ata Final")
        text_payload = summary_engine.finalize_summary_text.call_args.args[0]
        self.assertEqual(text_payload["title"], final_payload["title"])
        self.assertEqual(text_payload["room_name"], "talk__dev__roomA")
        self.assertEqual(text_payload["transcript_session_id"], "session-1")
        self.assertEqual(text_payload["call_session_id"], "RM_session-1")
        self.assertIsInstance(text_payload["updated_at"], str)
        self.assertTrue(text_payload["updated_at"])
        final_upload_payload = router.upload_json.call_args.args[2]
        self.assertEqual(final_upload_payload["summary"], final_payload)
        self.assertNotIn("summary", text_payload)

        publish_kwargs = router.publish_call_index.call_args.kwargs
        self.assertTrue(publish_kwargs["final_summary_ready"])
        self.assertTrue(publish_kwargs["final_summary_text_ready"])
        self.assertEqual(
            publish_kwargs["final_summary_text_path"],
            STT_APP.session_final_summary_text_path(routing.storage_base_path),
        )

    async def test_final_summary_text_failure_marks_task_error_for_retry(self):
        stop_event = asyncio.Event()
        routing = STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path="VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )
        final_payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo executivo objetivo.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }

        router = MagicMock()
        router.fetch_json.return_value = {"summary": STT_APP.default_accumulated_summary_payload()}
        router.fetch_agent_prompt.return_value = None
        router.upload_json = MagicMock()
        router.upload_text = MagicMock()
        router.publish_call_index = MagicMock()

        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.final_model = "final-model"
        summary_engine.final_text_model = "final-text-model"
        summary_engine.finalize_summary.return_value = final_payload
        summary_engine.finalize_summary_text.side_effect = RuntimeError("falha ata")

        task = {
            "room_name": "talk__dev__roomA",
            "call_session_id": "RM_session-1",
            "minute_index": -1,
            "retries": 0,
        }
        session_row = {"finalized_at": "2026-04-24T12:00:00+00:00"}
        final_task_rows = [
            {
                "minute_index": -1,
                "status": "processing",
                "retries": 0,
                "updated_at": "2026-04-24T12:00:00+00:00",
            }
        ]

        claim_calls = {"count": 0}

        def fake_claim_summary_task(_now_iso: str):
            if claim_calls["count"] == 0:
                claim_calls["count"] += 1
                return task
            stop_event.set()
            return None

        done_calls: list[tuple[str, str, int]] = []
        error_calls: list[tuple[str, str, int, int, str]] = []

        def fake_mark_done(room_name: str, call_session_id: str, minute_index: int, _now_iso: str):
            done_calls.append((room_name, call_session_id, minute_index))
            stop_event.set()

        def fake_mark_error(
            room_name: str,
            call_session_id: str,
            minute_index: int,
            retries: int,
            error_message: str,
            _now_iso: str,
        ):
            error_calls.append((room_name, call_session_id, minute_index, retries, error_message))
            stop_event.set()

        with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
            with patch.object(STT_APP, "db_claim_summary_task", side_effect=fake_claim_summary_task):
                with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
                    with patch.object(STT_APP, "routing_context_from_session_row", return_value=routing):
                        with patch.object(STT_APP, "db_get_summary_task_rows", return_value=final_task_rows):
                            with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                                with patch.object(STT_APP, "db_mark_summary_task_done", side_effect=fake_mark_done):
                                    with patch.object(
                                        STT_APP,
                                        "db_mark_summary_task_error",
                                        side_effect=fake_mark_error,
                                    ):
                                        await asyncio.wait_for(
                                            STT_APP.summary_worker_loop(stop_event, router, summary_engine),
                                            timeout=2,
                                        )

        self.assertEqual(done_calls, [])
        self.assertEqual(len(error_calls), 1)
        self.assertEqual(error_calls[0][:4], ("talk__dev__roomA", "RM_session-1", -1, 1))
        self.assertIn("falha ao gerar/enviar ata final textual", error_calls[0][4])
        router.upload_text.assert_not_called()
        publish_kwargs = router.publish_call_index.call_args.kwargs
        self.assertTrue(publish_kwargs["final_summary_ready"])
        self.assertFalse(publish_kwargs["final_summary_text_ready"])

    async def test_final_summary_text_retry_reuses_existing_final_summary_json(self):
        stop_event = asyncio.Event()
        routing = STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path="VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )
        final_payload = {
            "title": "Resumo Final Executivo da Chamada",
            "conversation_types": [],
            "executive_summary": "Resumo executivo objetivo.",
            "topics": [],
            "global_decisions": [],
            "global_pending_items": [],
            "global_next_steps": [],
            "additional_notes": [],
        }

        router = MagicMock()
        router.fetch_json.side_effect = [{"summary": STT_APP.default_accumulated_summary_payload()}, {"summary": final_payload}]
        router.fetch_agent_prompt.return_value = None
        router.upload_json = MagicMock()
        router.upload_text = MagicMock()
        router.publish_call_index = MagicMock()

        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.minute_model = "minute-model"
        summary_engine.accumulated_model = "accumulated-model"
        summary_engine.final_model = "final-model"
        summary_engine.final_text_model = "final-text-model"
        summary_engine.finalize_summary.return_value = {"unexpected": True}
        summary_engine.finalize_summary_text.return_value = "# Ata Final"

        task = {
            "room_name": "talk__dev__roomA",
            "call_session_id": "RM_session-1",
            "minute_index": -1,
            "retries": 1,
        }
        session_row = {"finalized_at": "2026-04-24T12:00:00+00:00"}
        final_task_rows = [
            {
                "minute_index": -1,
                "status": "processing",
                "retries": 1,
                "updated_at": "2026-04-24T12:00:00+00:00",
            }
        ]

        claim_calls = {"count": 0}

        def fake_claim_summary_task(_now_iso: str):
            if claim_calls["count"] == 0:
                claim_calls["count"] += 1
                return task
            stop_event.set()
            return None

        done_calls: list[tuple[str, str, int]] = []

        def fake_mark_done(room_name: str, call_session_id: str, minute_index: int, _now_iso: str):
            done_calls.append((room_name, call_session_id, minute_index))
            stop_event.set()

        with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
            with patch.object(STT_APP, "db_claim_summary_task", side_effect=fake_claim_summary_task):
                with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
                    with patch.object(STT_APP, "routing_context_from_session_row", return_value=routing):
                        with patch.object(STT_APP, "db_get_summary_task_rows", return_value=final_task_rows):
                            with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                                with patch.object(STT_APP, "db_mark_summary_task_done", side_effect=fake_mark_done):
                                    with patch.object(STT_APP, "db_mark_summary_task_error") as mark_error_mock:
                                        await asyncio.wait_for(
                                            STT_APP.summary_worker_loop(stop_event, router, summary_engine),
                                            timeout=2,
                                        )

        self.assertEqual(done_calls, [("talk__dev__roomA", "RM_session-1", -1)])
        mark_error_mock.assert_not_called()
        summary_engine.finalize_summary.assert_not_called()
        summary_engine.finalize_summary_text.assert_called_once()
        router.upload_text.assert_called_once()
        publish_kwargs = router.publish_call_index.call_args.kwargs
        self.assertTrue(publish_kwargs["final_summary_ready"])
        self.assertTrue(publish_kwargs["final_summary_text_ready"])

    async def test_summary_worker_logs_main_summary_steps(self):
        stop_event = asyncio.Event()
        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.provider = "openrouter"
        summary_engine.minute_model = "minute-model"
        summary_engine.accumulated_model = "accumulated-model"
        summary_engine.final_model = "final-model"
        summary_engine.final_text_model = "final-text-model"

        with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
            with patch.object(STT_APP, "db_claim_summary_task", side_effect=lambda _now_iso: (stop_event.set() or None)):
                with self.assertLogs("stt", level="INFO") as logs:
                    await asyncio.wait_for(
                        STT_APP.summary_worker_loop(stop_event, MagicMock(), summary_engine),
                        timeout=2,
                    )

        log_text = "\n".join(logs.output)
        self.assertIn("iniciando geração do sumario", log_text)
        self.assertIn("modelos: { minuto: minute-model, acumulado: accumulated-model, final: final-model, final_text: final-text-model }", log_text)

    async def test_summary_reconciliation_logs_zero_counts(self):
        with patch.object(STT_APP, "db_recover_stale_summary_tasks", return_value=0):
            with patch.object(STT_APP, "db_get_finalized_sessions_for_summary_reconcile", return_value=[]):
                with self.assertLogs("stt", level="INFO") as logs:
                    await STT_APP.run_summary_reconciliation_once()

        self.assertIn(
            "Reconciliando summaries: recovered=0 sessions=0 queued=0 reopened=0 late_requeued=0",
            "\n".join(logs.output),
        )


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class ProgressiveAtaSummaryTests(unittest.IsolatedAsyncioTestCase):
    def build_routing(self):
        return STT_APP.RoomRoutingContext(
            namespace="talk__dev",
            room_name="talk__dev__roomA",
            room_id="roomA",
            call_session_id="RM_session-1",
            transcript_session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path="VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
        )

    async def test_progressive_minute_updates_accumulated_text_and_meta(self):
        stop_event = asyncio.Event()
        routing = self.build_routing()
        transcript_path = f"{routing.storage_base_path}/minutes/0000/transcript.json"
        summary_text_path = STT_APP.minute_progressive_ata_summary_path(routing.storage_base_path, 0)

        router = MagicMock()
        router.fetch_json.return_value = {
            "minute_index": 0,
            "minute_started_at": "2026-04-24T12:00:00+00:00",
            "minute_ended_at": "2026-04-24T12:10:00+00:00",
            "lines": [{"speaker": "Daiane", "text": "Estou testando o fluxo de SLA."}],
        }
        router.fetch_text.return_value = ""
        router.upload_text = MagicMock()
        router.upload_json = MagicMock()
        router.publish_call_index = MagicMock()

        summary_engine = MagicMock()
        summary_engine.enabled = True
        summary_engine.accumulated_model = "acc-model"
        summary_engine.update_progressive_ata.return_value = "<h1>ATA - Daily</h1>"
        summary_engine.summarize_minute = MagicMock()
        summary_engine.merge_summaries = MagicMock()

        task = {
            "room_name": "talk__dev__roomA",
            "call_session_id": "RM_session-1",
            "minute_index": 0,
            "retries": 0,
        }
        export_row = {
            "minute_index": 0,
            "transcript_json_path": transcript_path,
            "summary_json_path": summary_text_path,
        }
        claim_calls = {"count": 0}

        def fake_claim_summary_task(_now_iso: str):
            if claim_calls["count"] == 0:
                claim_calls["count"] += 1
                return task
            stop_event.set()
            return None

        done_calls = []

        def fake_mark_done(room_name: str, call_session_id: str, minute_index: int, _now_iso: str):
            done_calls.append((room_name, call_session_id, minute_index))
            stop_event.set()

        with patch.object(STT_APP, "SUMMARY_MODE", "ata_progressiva"):
            with patch.object(STT_APP, "run_summary_reconciliation_once", new=AsyncMock()):
                with patch.object(STT_APP, "db_claim_summary_task", side_effect=fake_claim_summary_task):
                    with patch.object(STT_APP, "db_get_session_row", return_value={"finalized_at": None}):
                        with patch.object(STT_APP, "routing_context_from_session_row", return_value=routing):
                            with patch.object(STT_APP, "db_get_minute_export", return_value=export_row):
                                with patch.object(STT_APP, "db_update_minute_export_summary_path") as update_path:
                                    with patch.object(STT_APP, "db_mark_summary_task_done", side_effect=fake_mark_done):
                                        with patch.object(STT_APP, "db_mark_summary_task_error") as mark_error:
                                            await asyncio.wait_for(
                                                STT_APP.summary_worker_loop(stop_event, router, summary_engine),
                                                timeout=2,
                                            )

        self.assertEqual(done_calls, [("talk__dev__roomA", "RM_session-1", 0)])
        mark_error.assert_not_called()
        summary_engine.summarize_minute.assert_not_called()
        summary_engine.merge_summaries.assert_not_called()
        summary_engine.update_progressive_ata.assert_called_once()
        update_path.assert_called_once_with(
            "talk__dev__roomA",
            "RM_session-1",
            0,
            summary_text_path,
            ANY,
        )
        uploaded_text_paths = [call.args[1] for call in router.upload_text.call_args_list]
        self.assertIn(summary_text_path, uploaded_text_paths)
        self.assertIn(
            STT_APP.session_progressive_ata_accumulated_path(routing.storage_base_path),
            uploaded_text_paths,
        )
        meta_upload = router.upload_json.call_args.args
        self.assertEqual(meta_upload[1], STT_APP.session_progressive_ata_meta_path(routing.storage_base_path))
        self.assertEqual(meta_upload[2]["last_minute_index"], 0)
        self.assertEqual(
            router.publish_call_index.call_args.kwargs["summary_accumulated_path"],
            STT_APP.session_progressive_ata_accumulated_path(routing.storage_base_path),
        )

    async def test_progressive_final_source_policy(self):
        routing = self.build_routing()
        exports = [
            {
                "minute_index": 0,
                "transcript_json_path": f"{routing.storage_base_path}/minutes/0000/transcript.json",
            }
        ]
        task_rows = [
            {"minute_index": 0, "status": "done"},
            {"minute_index": -1, "status": "processing"},
        ]

        async def run_case(requested_source: str, max_chars: int, expected_mode: str):
            router = MagicMock()

            def fake_fetch_json(_routing, path):
                if path == STT_APP.session_progressive_ata_meta_path(routing.storage_base_path):
                    return {"last_minute_index": -1}
                if path == STT_APP.session_final_transcript_path(routing.storage_base_path):
                    return {
                        "lines": [
                            {"speaker": "A", "text": "x" * 30},
                            {"speaker": "B", "text": "y" * 30},
                        ]
                    }
                return {"lines": [{"speaker": "A", "text": "delta"}]}

            router.fetch_json.side_effect = fake_fetch_json
            router.fetch_text.return_value = "<h1>ATA - Daily</h1>"
            router.upload_text = MagicMock()
            router.publish_call_index = MagicMock()

            summary_engine = MagicMock()
            summary_engine.final_text_model = "final-text-model"
            summary_engine.finalize_progressive_ata.return_value = "<h1>ATA Final</h1>"

            with patch.object(STT_APP, "SUMMARY_MODE", "ata_progressiva"):
                with patch.object(STT_APP, "SUMMARY_PROGRESSIVE_FINAL_SOURCE", requested_source):
                    with patch.object(STT_APP, "SUMMARY_PROGRESSIVE_FULL_TRANSCRIPT_MAX_CHARS", max_chars):
                        with patch.object(STT_APP, "db_get_summary_task_rows", return_value=task_rows):
                            with patch.object(STT_APP, "db_get_session_minute_exports", return_value=exports):
                                processed = await STT_APP.process_progressive_ata_final_task(
                                    router,
                                    summary_engine,
                                    routing,
                                    {"finalized_at": "2026-04-24T12:00:00+00:00"},
                                    "talk__dev__roomA",
                                    "RM_session-1",
                                    -1,
                                    "2026-04-24T12:10:00+00:00",
                                )

            self.assertTrue(processed)
            call_args = summary_engine.finalize_progressive_ata.call_args.args
            self.assertEqual(call_args[2], expected_mode)
            return call_args[1]

        full_text = await run_case("auto", 1000, "full_transcript")
        self.assertIn("x" * 30, full_text)

        delta_text = await run_case("auto", 10, "delta_only")
        self.assertIn("delta", delta_text)
        self.assertNotIn("x" * 30, delta_text)

        forced_delta_text = await run_case("delta_only", 1000, "delta_only")
        self.assertIn("delta", forced_delta_text)

        forced_full_text = await run_case("full_transcript", 10, "full_transcript")
        self.assertIn("x" * 30, forced_full_text)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class FinalTranscriptTests(unittest.TestCase):
    def test_session_final_transcript_path_matches_expected_structure(self):
        base = "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1"
        self.assertEqual(
            STT_APP.session_final_transcript_path(base),
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1/final/final_transcript.json",
        )

    def test_session_final_summary_text_path_matches_expected_structure(self):
        base = "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1"
        self.assertEqual(
            STT_APP.session_final_summary_text_path(base),
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1/final/final_summary_text.txt",
        )

    def test_build_final_transcript_payload_uses_room_aggregate(self):
        lines = [
            {
                "speaker": "Alice",
                "text": "Oi",
                "chunk_started_at": "2026-04-18T10:00:00+00:00",
                "chunk_ended_at": "2026-04-18T10:00:02+00:00",
            }
        ]
        with patch.object(STT_APP, "db_get_room_aggregate", return_value=("[Alice] Oi", lines)):
            payload = STT_APP.build_final_transcript_payload(
                room_name="talk__dev__roomA",
                call_session_id="RM_session-1",
                transcript_session_id="session-1",
                now_iso="2026-04-18T10:10:00+00:00",
            )

        self.assertEqual(payload["transcript"], "[Alice] Oi")
        self.assertEqual(payload["lines"], lines)
        self.assertEqual(payload["line_count"], 1)
        self.assertEqual(payload["updated_at"], "2026-04-18T10:10:00+00:00")

    def test_db_get_room_aggregate_preserves_order_speaker_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    conn = STT_APP.read_db_connection()
                    try:
                        now = "2026-04-18T10:00:00+00:00"
                        conn.execute(
                            """
                            INSERT INTO sessions(
                                room_name, session_id, started_at,
                                state, room_end_received, last_chunk_at, created_at, updated_at
                            ) VALUES (?, ?, ?, 'active', 0, ?, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", now, now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO participants(
                                room_name, session_id, participant_identity, participant_name,
                                started_at, state, last_seq, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", "user_a", "Alice", now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO participants(
                                room_name, session_id, participant_identity, participant_name,
                                started_at, state, last_seq, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", "user_b", "Bruno", now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO chunks(
                                room_name, session_id, participant_identity, seq, track_sid,
                                chunk_started_at, chunk_ended_at, sample_rate, channels, encoding,
                                spool_path, status, transcript, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                "user_b",
                                1,
                                "track-b",
                                "2026-04-18T10:00:05+00:00",
                                "2026-04-18T10:00:06+00:00",
                                16000,
                                1,
                                "pcm_s16le",
                                "/tmp/chunk-b1.wav",
                                "done",
                                "Bom dia",
                                now,
                                now,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO chunks(
                                room_name, session_id, participant_identity, seq, track_sid,
                                chunk_started_at, chunk_ended_at, sample_rate, channels, encoding,
                                spool_path, status, transcript, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                "user_a",
                                1,
                                "track-a",
                                "2026-04-18T10:00:05+00:00",
                                "2026-04-18T10:00:06+00:00",
                                16000,
                                1,
                                "pcm_s16le",
                                "/tmp/chunk-a1.wav",
                                "done",
                                "Oi",
                                now,
                                now,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO chunks(
                                room_name, session_id, participant_identity, seq, track_sid,
                                chunk_started_at, chunk_ended_at, sample_rate, channels, encoding,
                                spool_path, status, transcript, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                "user_b",
                                2,
                                "track-b",
                                "2026-04-18T10:00:08+00:00",
                                "2026-04-18T10:00:09+00:00",
                                16000,
                                1,
                                "pcm_s16le",
                                "/tmp/chunk-b2.wav",
                                "done",
                                "Tudo certo",
                                now,
                                now,
                            ),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    transcript, lines = STT_APP.db_get_room_aggregate("talk__dev__roomA", "RM_session-1")

        self.assertEqual([line["speaker"] for line in lines], ["Alice", "Bruno", "Bruno"])
        self.assertEqual([line["seq"] for line in lines], [1, 1, 2])
        self.assertIn("chunk_started_at", lines[0])
        self.assertIn("chunk_ended_at", lines[0])
        self.assertEqual(transcript, "[Alice] Oi\n[Bruno] Bom dia\n[Bruno] Tudo certo")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class AdminSummaryReprocessTests(unittest.IsolatedAsyncioTestCase):
    def build_session_row(self, *, finalized_at: str | None = "2026-04-24T12:00:00+00:00") -> dict:
        return {
            "room_name": "talk__dev__roomA",
            "session_id": "RM_session-1",
            "call_session_id": "RM_session-1",
            "transcript_session_id": "session-1",
            "room_id": "roomA",
            "vertical": "HEALTH",
            "slug": "acme",
            "firestore_doc_path": "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/SESSIONS/RM_session-1",
            "storage_base_path": "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1",
            "started_at": "2026-04-24T10:00:00+00:00",
            "room_end_received": 1,
            "finalized_at": finalized_at,
        }

    async def test_admin_reprocess_post_accepts_finalized_session(self):
        payload = STT_APP.AdminSummaryReprocessTarget(
            namespace="talk__dev",
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            call_session_id="RM_session-1",
        )
        session_row = self.build_session_row()
        runtime = SimpleNamespace(
            summary_engine=SimpleNamespace(enabled=True),
            firebase_router=SimpleNamespace(
                upload_json=MagicMock(),
                publish_call_index=MagicMock(),
            ),
        )
        STT_APP.app.state.runtime = runtime

        task_rows = [
            {
                "minute_index": 0,
                "status": "pending",
                "retries": 0,
                "error_message": None,
            },
            {
                "minute_index": -1,
                "status": "pending",
                "retries": 0,
                "error_message": None,
            },
        ]
        minute_exports = [{"minute_index": 0}]

        with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
            with patch.object(STT_APP, "publish_session_minute_exports", return_value=0):
                with patch.object(
                    STT_APP, "db_reset_summary_reprocess_state", return_value={"minute_exports": 1}
                ):
                    with patch.object(
                        STT_APP, "build_final_transcript_payload", return_value={"line_count": 1}
                    ):
                        with patch.object(
                            STT_APP, "db_get_session_minute_exports", return_value=minute_exports
                        ):
                            with patch.object(STT_APP, "db_get_summary_task_rows", return_value=task_rows):
                                with patch.object(
                                    STT_APP, "db_force_finalize_session_for_reprocess"
                                ) as force_finalize_mock:
                                    response = await STT_APP.admin_summary_reprocess(payload)

        self.assertEqual(response.status_code, 202)
        data = json.loads(response.body)
        self.assertEqual(data["overall_status"], "pending")
        self.assertEqual(data["session"]["room_name"], "talk__dev__roomA")
        force_finalize_mock.assert_not_called()

    async def test_admin_reprocess_post_forces_finalize_when_session_active(self):
        payload = STT_APP.AdminSummaryReprocessTarget(
            namespace="talk__dev",
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            call_session_id="RM_session-1",
        )
        session_row_active = self.build_session_row(finalized_at=None)
        session_row_finalized = self.build_session_row()
        runtime = SimpleNamespace(
            summary_engine=SimpleNamespace(enabled=True),
            firebase_router=SimpleNamespace(
                upload_json=MagicMock(),
                publish_call_index=MagicMock(),
            ),
        )
        STT_APP.app.state.runtime = runtime

        with patch.object(
            STT_APP,
            "db_get_session_row",
            side_effect=[session_row_active, session_row_finalized],
        ):
            with patch.object(STT_APP, "publish_session_minute_exports", return_value=0):
                with patch.object(
                    STT_APP, "db_reset_summary_reprocess_state", return_value={"minute_exports": 0}
                ):
                    with patch.object(
                        STT_APP, "build_final_transcript_payload", return_value={"line_count": 0}
                    ):
                        with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                            with patch.object(STT_APP, "db_get_summary_task_rows", return_value=[]):
                                with patch.object(
                                    STT_APP, "db_force_finalize_session_for_reprocess"
                                ) as force_finalize_mock:
                                    response = await STT_APP.admin_summary_reprocess(payload)

        self.assertEqual(response.status_code, 202)
        force_finalize_mock.assert_called_once()

    async def test_admin_reprocess_post_rejects_target_mismatch(self):
        payload = STT_APP.AdminSummaryReprocessTarget(
            namespace="talk__dev",
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            call_session_id="RM_session-1",
        )
        session_row = self.build_session_row()
        session_row["slug"] = "other"
        runtime = SimpleNamespace(summary_engine=SimpleNamespace(enabled=True), firebase_router=MagicMock())
        STT_APP.app.state.runtime = runtime

        with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
            with self.assertRaises(STT_APP.HTTPException) as ctx:
                await STT_APP.admin_summary_reprocess(payload)
        self.assertEqual(ctx.exception.status_code, 409)

    async def test_admin_reprocess_status_returns_final_summary_when_done(self):
        runtime = SimpleNamespace(
            firebase_router=SimpleNamespace(fetch_json=MagicMock(return_value={"summary": {"title": "ok"}}))
        )
        STT_APP.app.state.runtime = runtime
        session_row = self.build_session_row()
        task_rows = [
            {
                "minute_index": 0,
                "status": "done",
                "retries": 0,
                "error_message": None,
            },
            {
                "minute_index": -1,
                "status": "done",
                "retries": 0,
                "error_message": None,
            },
        ]
        minute_exports = [{"minute_index": 0}]

        with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
            with patch.object(STT_APP, "db_get_summary_task_rows", return_value=task_rows):
                with patch.object(STT_APP, "db_get_session_minute_exports", return_value=minute_exports):
                    response = await STT_APP.admin_summary_reprocess_status(
                        namespace="talk__dev",
                        vertical="HEALTH",
                        slug="acme",
                        room_id="roomA",
                        call_session_id="RM_session-1",
                    )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["overall_status"], "done")
        self.assertEqual(data["final_summary"]["title"], "ok")

    async def test_admin_reprocess_status_returns_final_error_details_when_exhausted(self):
        temp_payload = {
            "error": "falha de contrato",
            "model": "openai/gpt-5.6-luna",
            "kind": STT_APP.SUMMARY_KIND_FINAL,
            "updated_at": "2026-04-24T12:00:00+00:00",
        }
        runtime = SimpleNamespace(
            firebase_router=SimpleNamespace(fetch_json=MagicMock(return_value=temp_payload))
        )
        STT_APP.app.state.runtime = runtime
        session_row = self.build_session_row()
        task_rows = [
            {
                "minute_index": -1,
                "status": "error",
                "retries": STT_APP.SUMMARY_MAX_RETRIES,
                "error_message": "exhausted",
            }
        ]

        with patch.object(STT_APP, "db_get_session_row", return_value=session_row):
            with patch.object(STT_APP, "db_get_summary_task_rows", return_value=task_rows):
                with patch.object(STT_APP, "db_get_session_minute_exports", return_value=[]):
                    response = await STT_APP.admin_summary_reprocess_status(
                        namespace="talk__dev",
                        vertical="HEALTH",
                        slug="acme",
                        room_id="roomA",
                        call_session_id="RM_session-1",
                    )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.body)
        self.assertEqual(data["overall_status"], "error_exhausted")
        self.assertEqual(data["final_error_details"]["error"], "falha de contrato")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryReprocessDbTests(unittest.TestCase):
    def test_db_reset_summary_reprocess_state_resets_flags_and_summary_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    now = "2026-04-24T12:00:00+00:00"
                    conn = STT_APP.read_db_connection()
                    try:
                        conn.execute(
                            """
                            INSERT INTO minute_exports(
                                room_name, session_id, minute_index,
                                transcript_json_path, summary_json_path, content_hash,
                                minute_started_at, minute_ended_at, finalized,
                                exported_at, updated_at
                            ) VALUES (?, ?, 0, ?, ?, ?, ?, ?, 1, ?, ?)
                            """,
                            (
                                "talk__dev__roomA",
                                "RM_session-1",
                                "minutes/0000/transcript.json",
                                "minutes/0000/summary.json",
                                "h1",
                                now,
                                now,
                                now,
                                now,
                            ),
                        )
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, 0, 'error', 2, ?, 'err', ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, -1, 'done', 1, ?, NULL, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", now, now, now),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    STT_APP.db_reset_summary_reprocess_state("talk__dev__roomA", "RM_session-1", now)
                    rows = STT_APP.db_get_summary_task_rows("talk__dev__roomA", "RM_session-1")
                    minute_row = next(row for row in rows if int(row["minute_index"]) == 0)
                    final_row = next(row for row in rows if int(row["minute_index"]) < 0)
                    self.assertEqual(minute_row["status"], "pending")
                    self.assertEqual(int(minute_row["retries"]), 0)
                    self.assertEqual(final_row["status"], "pending")
                    self.assertEqual(int(final_row["retries"]), 0)

                    export_row = STT_APP.db_get_minute_export("talk__dev__roomA", "RM_session-1", 0)
                    self.assertIsNone(export_row["summary_json_path"])

    def test_db_claim_summary_task_prioritizes_minutes_before_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "queue.db"
            spool_dir = Path(tmpdir) / "spool"
            with patch.object(STT_APP, "SQLITE_PATH", db_path):
                with patch.object(STT_APP, "SPOOL_DIR", spool_dir):
                    STT_APP.init_db()
                    now = "2026-04-24T12:00:00+00:00"
                    conn = STT_APP.read_db_connection()
                    try:
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", 5, now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", 1, now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO summary_tasks(
                                room_name, session_id, minute_index,
                                status, retries, next_attempt_at,
                                error_message, created_at, updated_at
                            ) VALUES (?, ?, -1, 'pending', 0, ?, NULL, ?, ?)
                            """,
                            ("talk__dev__roomA", "RM_session-1", now, now, now),
                        )
                        conn.commit()
                    finally:
                        conn.close()

                    first = STT_APP.db_claim_summary_task(now)
                    second = STT_APP.db_claim_summary_task(now)
                    third = STT_APP.db_claim_summary_task(now)

        self.assertEqual(int(first["minute_index"]), 1)
        self.assertEqual(int(second["minute_index"]), 5)
        self.assertEqual(int(third["minute_index"]), -1)


if __name__ == "__main__":
    unittest.main()
