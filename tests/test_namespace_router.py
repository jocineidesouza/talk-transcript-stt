import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


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

    def test_request_text_retries_and_succeeds_after_timeouts(self):
        engine = self.build_engine()
        success_payload = {"output_text": "{}"}
        with patch.object(
            STT_APP.urllib.request,
            "urlopen",
            side_effect=[TimeoutError("timeout-1"), TimeoutError("timeout-2"), self._response(success_payload)],
        ) as urlopen_mock:
            with patch.object(STT_APP.time, "sleep") as sleep_mock:
                raw = engine._request_text("gpt-4.1-mini", "system", "user")

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
                    engine._request_text("gpt-4.1-mini", "system", "user")

        self.assertEqual(urlopen_mock.call_count, 3)
        self.assertEqual(sleep_mock.call_count, 2)


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
                }
            ],
        }
        normalized = STT_APP.validate_minute_summary_payload(payload)
        self.assertEqual(normalized["notes"][0]["confidence"], "medium")

    def test_validate_minute_summary_payload_normalizes_invalid_hypotheses_confidence(self):
        payload = {
            "chunk_type": "mista",
            "facts": [],
            "hypotheses": [
                {
                    "text": "hipotese inicial",
                    "confidence": "unknown",
                    "status": "uncertain",
                    "tags": ["hipotese"],
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
