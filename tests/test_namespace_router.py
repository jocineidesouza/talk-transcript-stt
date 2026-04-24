import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


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
            api_key="test-key",
            minute_model="minute-model",
            accumulated_model="accumulated-model",
            final_model="final-model",
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
                "title": "Resumo Final Executivo da Chamada",
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
        self.assertEqual(parsed["title"], "Resumo Final Executivo da Chamada")
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
                }
                for index in range(21)
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
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

    def test_validate_accumulated_summary_payload_accepts_topics_with_item_reference(self):
        payload = {
            "conversation_types": ["mista"],
            "topics": [
                {
                    "name": "Escopo do projeto",
                    "summary": "Escopo consolidado da fase inicial.",
                    "status": "active",
                    "tags": ["escopo"],
                }
            ],
            "facts": [
                {
                    "text": "Escopo da fase 1 foi confirmado.",
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
            "notes": [],
        }
        normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(normalized["topics"][0]["status"], "active")

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
    def _item(self, text: str, status: str, confidence: str = "medium", topic: str = "topico_1") -> dict:
        return {
            "text": text,
            "confidence": confidence,
            "status": status,
            "tags": ["tag"],
            "topic": topic,
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
                }
            ],
            "facts": [self._item(f"fact {i}", "confirmed", "high") for i in range(facts_count)],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [self._item(f"note {i}", "info") for i in range(notes_count)],
        }

    def test_validate_accumulated_summary_accepts_40_notes(self):
        with patch.object(STT_APP, "OPENAI_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(notes_count=40)
            normalized = STT_APP.validate_accumulated_summary_payload(payload)
        self.assertEqual(len(normalized["notes"]), 40)

    def test_validate_accumulated_summary_rejects_41_notes(self):
        with patch.object(STT_APP, "OPENAI_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(notes_count=41)
            with self.assertRaises(RuntimeError):
                STT_APP.validate_accumulated_summary_payload(payload)

    def test_validate_accumulated_summary_rejects_41_facts(self):
        with patch.object(STT_APP, "OPENAI_ACCUMULATED_MAX_ITEMS", 40):
            payload = self._payload(facts_count=41)
            with self.assertRaises(RuntimeError):
                STT_APP.validate_accumulated_summary_payload(payload)


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class SummaryEngineRetryTests(unittest.TestCase):
    def build_engine(self) -> object:
        return STT_APP.SummaryEngine(
            enabled=True,
            api_key="test-key",
            minute_model="minute-model",
            accumulated_model="accumulated-model",
            final_model="final-model",
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
            "facts": [],
            "hypotheses": [],
            "decisions": [],
            "open_items": [],
            "next_steps": [],
            "notes": [],
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
                raw = engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "gpt-4.1-mini", "system", "user")

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
                    engine._request_text(STT_APP.SUMMARY_KIND_MINUTE, "gpt-4.1-mini", "system", "user")

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
class OpenAISecretTests(unittest.TestCase):
    def test_load_openai_api_key_returns_key_from_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            secret_path = Path(tmpdir) / "openai_apikey.json"
            secret_path.write_text(json.dumps({"api_key": "test-key"}), encoding="utf-8")
            with patch.object(STT_APP, "OPENAI_SUMMARY_ENABLED", True):
                with patch.object(STT_APP, "OPENAI_APIKEY_FILE", secret_path):
                    self.assertEqual(STT_APP.load_openai_api_key(), "test-key")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class ConfigDefaultsTests(unittest.TestCase):
    def test_openai_request_timeout_default_is_300_in_source(self):
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("OPENAI_REQUEST_TIMEOUT_SECONDS", "300")', source)


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
        )

        payload = fake_doc.set.call_args.args[0]
        self.assertEqual(payload["status"], "finalized")
        self.assertTrue(payload["finalized"])
        self.assertEqual(payload["final_summary_path"], "path/final_summary.json")
        self.assertTrue(payload["final_summary_ready"])
        self.assertEqual(payload["final_transcript_path"], "path/final_transcript.json")
        self.assertTrue(payload["final_transcript_ready"])

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
        self.assertEqual(accumulated["facts"][0]["name"], "Planejamento")

    def test_build_deterministic_final_summary_includes_partial_disclosure(self):
        accumulated = {
            "conversation_types": ["executiva"],
            "topics": [
                {
                    "name": "Cronograma",
                    "summary": "Discussao do cronograma.",
                    "status": "active",
                    "tags": ["prazo"],
                }
            ],
            "facts": [],
            "hypotheses": [],
            "decisions": [
                {
                    "text": "Follow-up marcado para sexta.",
                    "confidence": "high",
                    "status": "confirmed",
                    "tags": ["followup"],
                    "name": "Cronograma",
                }
            ],
            "open_items": [],
            "next_steps": [],
            "notes": [],
        }
        final_payload = STT_APP.build_deterministic_final_summary(
            accumulated,
            [2, 5],
            ["acumulado reconstruido"],
        )
        self.assertIn("Documento parcial", final_payload["executive_summary"])
        self.assertTrue(
            any("Ata parcial" in note["text"] for note in final_payload["additional_notes"])
        )

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


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class FinalTranscriptTests(unittest.TestCase):
    def test_session_final_transcript_path_matches_expected_structure(self):
        base = "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1"
        self.assertEqual(
            STT_APP.session_final_transcript_path(base),
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/RM_session-1/final/final_transcript.json",
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


if __name__ == "__main__":
    unittest.main()
