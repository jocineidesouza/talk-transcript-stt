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

        with patch.object(router, "sink_for_room", return_value=disabled_sink):
            router.publish_call_index(
                room_name="talk__dev__roomA",
                session_id="session-1",
                status="processing",
                last_minute_index=3,
                finalized=False,
            )

        disabled_sink.publish_call_index.assert_called_once()


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


if __name__ == "__main__":
    unittest.main()
