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
            session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path=(
                "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/TRANSCRIPT/session-1"
            ),
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1",
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
            routing = router.resolve_room_routing_context("talk__dev__roomA", "session-1")

        self.assertEqual(routing.namespace, "talk__dev")
        self.assertEqual(routing.room_id, "roomA")
        self.assertEqual(routing.vertical, "HEALTH")
        self.assertEqual(routing.slug, "acme")
        self.assertEqual(
            routing.firestore_doc_path,
            "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/TRANSCRIPT/session-1",
        )
        self.assertEqual(
            routing.storage_base_path,
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1",
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
                router.resolve_room_routing_context("talk__dev__roomA", "session-1")

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
                router.resolve_room_routing_context("talk__dev__roomA", "session-1")

        self.assertEqual(ctx.exception.code, "room_index_invalid")


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class RoutingPathTests(unittest.TestCase):
    def test_build_paths_matches_expected_structure(self):
        doc_path = STT_APP.build_firestore_doc_path(
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            session_id="session-1",
        )
        storage_base = STT_APP.build_storage_base_path(
            vertical="HEALTH",
            slug="acme",
            room_id="roomA",
            session_id="session-1",
        )
        self.assertEqual(
            doc_path,
            "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/TRANSCRIPT/session-1",
        )
        self.assertEqual(
            storage_base,
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1",
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
            session_id="session-1",
            vertical="HEALTH",
            slug="acme",
            firestore_doc_path=(
                "VERTICALS/HEALTH/COMPANIES/acme/ROOMS/roomA/TRANSCRIPT/session-1"
            ),
            storage_base_path="VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1",
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


@unittest.skipIf(STT_APP_IMPORT_ERROR is not None, f"dependencias ausentes: {STT_APP_IMPORT_ERROR}")
class FinalTranscriptTests(unittest.TestCase):
    def test_session_final_transcript_path_matches_expected_structure(self):
        base = "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1"
        self.assertEqual(
            STT_APP.session_final_transcript_path(base),
            "VERTICALS/HEALTH/COMPANIES/acme/TRANSCRIPT/roomA/session-1/final/final_transcript.json",
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
                session_id="session-1",
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
                            ("talk__dev__roomA", "session-1", now, now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO participants(
                                room_name, session_id, participant_identity, participant_name,
                                started_at, state, last_seq, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
                            """,
                            ("talk__dev__roomA", "session-1", "user_a", "Alice", now, now, now),
                        )
                        conn.execute(
                            """
                            INSERT INTO participants(
                                room_name, session_id, participant_identity, participant_name,
                                started_at, state, last_seq, created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, 'active', 0, ?, ?)
                            """,
                            ("talk__dev__roomA", "session-1", "user_b", "Bruno", now, now, now),
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
                                "session-1",
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
                                "session-1",
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
                                "session-1",
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

                    transcript, lines = STT_APP.db_get_room_aggregate("talk__dev__roomA", "session-1")

        self.assertEqual([line["speaker"] for line in lines], ["Alice", "Bruno", "Bruno"])
        self.assertEqual([line["seq"] for line in lines], [1, 1, 2])
        self.assertIn("chunk_started_at", lines[0])
        self.assertIn("chunk_ended_at", lines[0])
        self.assertEqual(transcript, "[Alice] Oi\n[Bruno] Bom dia\n[Bruno] Tudo certo")


if __name__ == "__main__":
    unittest.main()
