import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, call, patch

import requests

from validate_orchid_dataset import (
    API_URL,
    MAX_REQUEST_ATTEMPTS,
    REQUEST_TIMEOUT,
    identify_orchid,
)


class IdentifyOrchidRetryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.image_path = Path(self.temp_dir.name) / "orchid.jpg"
        self.image_path.write_bytes(b"test-image")
        self.response = Mock(ok=True, json=lambda: {"results": []})

    @patch("validate_orchid_dataset.time.sleep")
    @patch("validate_orchid_dataset.requests.post")
    def test_retries_timeout_with_reopened_file_and_longer_timeout(
        self, post, sleep
    ):
        post.side_effect = [requests.Timeout("read timed out"), self.response]

        result = identify_orchid(self.image_path, "test-key")

        self.assertEqual(result, {"results": []})
        self.assertEqual(post.call_count, 2)
        self.assertEqual(sleep.call_args_list, [call(2)])
        first_files = post.call_args_list[0].kwargs["files"]
        second_files = post.call_args_list[1].kwargs["files"]
        first_stream = first_files[0][1][1]
        second_stream = second_files[0][1][1]
        self.assertIsNot(first_stream, second_stream)
        self.assertTrue(first_stream.closed)
        self.assertTrue(second_stream.closed)
        for request in post.call_args_list:
            self.assertEqual(request.args[0], API_URL)
            self.assertEqual(request.kwargs["timeout"], REQUEST_TIMEOUT)
            self.assertEqual(request.kwargs["params"], {"api-key": "test-key"})

    @patch("validate_orchid_dataset.time.sleep")
    @patch("validate_orchid_dataset.requests.post")
    def test_retries_connection_error_up_to_limit_then_omits_api_key(
        self, post, sleep
    ):
        post.side_effect = requests.ConnectionError(
            "failed request to https://example.invalid/?api-key=secret"
        )

        with self.assertRaises(RuntimeError) as context:
            identify_orchid(self.image_path, "secret")

        self.assertEqual(post.call_count, MAX_REQUEST_ATTEMPTS)
        self.assertEqual(sleep.call_args_list, [call(2), call(4)])
        self.assertNotIn("secret", str(context.exception))
        self.assertIn("ConnectionError", str(context.exception))


if __name__ == "__main__":
    unittest.main()
