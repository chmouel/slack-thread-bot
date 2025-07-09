import unittest
from unittest.mock import MagicMock, patch, call
import os
import logging
from main import SlackThreadBot, DEFAULT_TIMEZONES


class TestSlackThreadBot(unittest.TestCase):
    """Unit tests for the SlackThreadBot."""

    @classmethod
    def setUpClass(cls):
        """Disable logging for tests."""
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """Enable logging after tests."""
        logging.disable(logging.NOTSET)

    def setUp(self):
        """Set up the test environment before each test."""
        with patch("main.App"):
            self.bot = SlackThreadBot(
                bot_token="fake-bot-token", app_token="fake-app-token"
            )
            self.mock_client = MagicMock()
            self.bot.app.client = self.mock_client
            self.mock_client.conversations_open.return_value = {
                "channel": {"id": "D12345"}
            }

    def _create_message(self, text):
        """Helper to create a mock message payload."""
        return {
            "user": "U12345",
            "channel": "C12345",
            "ts": "1628630400.000200",
            "thread_ts": "1628630400.000200",
            "text": text,
        }

    @patch("main.subprocess.run")
    @patch("main.AirportTime", side_effect=ValueError("Invalid Code"))
    def test_handle_tz_no_codes(self, mock_airport_time, mock_subprocess_run):
        """Test !tz with a time string but no airport codes."""
        mock_subprocess_run.return_value.stdout = "Converted Time"
        message = self._create_message("!tz 10:30 tomorrow")

        with patch.dict(
            os.environ,
            {"BATZ_TIMEZONES": "Paris/Europe/Paris,Tokyo/Asia/Tokyo"},
            clear=True,
        ):
            self.bot.handle_tz(message, self.mock_client)

        self.assertEqual(mock_subprocess_run.call_count, 2)
        self.assertIn(
            "10:30 tomorrow", mock_subprocess_run.call_args_list[0].args[0][1]
        )
        mock_airport_time.assert_not_called()
        self.mock_client.chat_postMessage.assert_called_once()

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_single_code(self, mock_airport_time, mock_subprocess_run):
        """Test !tz with a time and a single airport code."""
        message = self._create_message("!tz 9:00am JFK")
        mock_airport_instance = MagicMock()
        setattr(mock_airport_instance.airport, "timezone", "America/New_York")
        mock_airport_time.return_value = mock_airport_instance
        mock_subprocess_run.return_value.stdout = "Converted Time"

        self.bot.handle_tz(message, self.mock_client)

        mock_airport_time.assert_called_once_with("JFK")
        mock_subprocess_run.assert_called_once()
        self.assertIn("9:00am", mock_subprocess_run.call_args.args[0][1])
        self.mock_client.chat_postMessage.assert_called_once()

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_multiple_codes_comma_separated(
        self, mock_airport_time, mock_subprocess_run
    ):
        """Test !tz with multiple comma-separated airport codes."""
        message = self._create_message("!tz 2pm LHR,CDG")

        def airport_time_side_effect(code):
            instance = MagicMock()
            if code == "LHR":
                setattr(instance.airport, "timezone", "Europe/London")
            elif code == "CDG":
                setattr(instance.airport, "timezone", "Europe/Paris")
            return instance

        mock_airport_time.side_effect = airport_time_side_effect
        mock_subprocess_run.return_value.stdout = "Converted Time"

        self.bot.handle_tz(message, self.mock_client)

        self.assertEqual(mock_airport_time.call_count, 2)
        mock_airport_time.assert_has_calls([call("LHR"), call("CDG")], any_order=True)
        self.assertIn("2pm", mock_subprocess_run.call_args.args[0][1])
        self.mock_client.chat_postMessage.assert_called_once()

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_multiple_codes_space_separated(
        self, mock_airport_time, mock_subprocess_run
    ):
        """Test !tz with multiple space-separated airport codes."""
        message = self._create_message("!tz next friday 18:00 SFO LAX")

        def airport_time_side_effect(code):
            instance = MagicMock()
            if code == "SFO":
                setattr(instance.airport, "timezone", "America/Los_Angeles")
            elif code == "LAX":
                setattr(instance.airport, "timezone", "America/Los_Angeles")
            return instance

        mock_airport_time.side_effect = airport_time_side_effect
        mock_subprocess_run.return_value.stdout = "Converted Time"

        self.bot.handle_tz(message, self.mock_client)

        self.assertEqual(mock_airport_time.call_count, 2)
        mock_airport_time.assert_has_calls([call("SFO"), call("LAX")], any_order=True)
        self.assertIn("next friday 18:00", mock_subprocess_run.call_args.args[0][1])
        self.mock_client.chat_postMessage.assert_called_once()

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_invalid_and_valid_codes(
        self, mock_airport_time, mock_subprocess_run
    ):
        """Test !tz with a mix of valid and invalid airport codes."""
        message = self._create_message("!tz 10:00 XXX,JFK,YYY")

        def airport_time_side_effect(code):
            if code == "JFK":
                instance = MagicMock()
                setattr(instance.airport, "timezone", "America/New_York")
                return instance
            raise ValueError("Invalid Code")

        mock_airport_time.side_effect = airport_time_side_effect
        mock_subprocess_run.return_value.stdout = "Converted Time"

        self.bot.handle_tz(message, self.mock_client)

        self.assertEqual(mock_airport_time.call_count, 3)

        self.mock_client.conversations_open.assert_called_once_with(users="U12345")
        self.mock_client.chat_postMessage.assert_any_call(
            channel="D12345",
            text="❌ Invalid airport code(s): XXX, YYY. Please use valid IATA airport codes.",
        )

        mock_subprocess_run.assert_called_once()
        self.assertIn("10:00", mock_subprocess_run.call_args.args[0][1])
        self.assertTrue(self.mock_client.chat_postMessage.call_count >= 2)

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_only_invalid_codes(self, mock_airport_time, mock_subprocess_run):
        """Test !tz with only invalid airport codes."""
        message = self._create_message("!tz 10:00 ZZZ,YYY")
        mock_airport_time.side_effect = ValueError("Invalid Code")

        self.bot.handle_tz(message, self.mock_client)

        self.mock_client.conversations_open.assert_called_once_with(users="U12345")
        self.mock_client.chat_postMessage.assert_called_once_with(
            channel="D12345",
            text="❌ Invalid airport code(s): ZZZ, YYY. Please use valid IATA airport codes.",
        )
        mock_subprocess_run.assert_not_called()

    @patch("main.subprocess.run")
    @patch("main.AirportTime")
    def test_handle_tz_no_args(self, mock_airport_time, mock_subprocess_run):
        """Test !tz with no arguments."""
        message = self._create_message("!tz")
        mock_subprocess_run.return_value.stdout = "Converted Time"

        # Convert tuple to a comma-separated string for the environment variable
        default_tz_str = ",".join(DEFAULT_TIMEZONES)
        with patch.dict(os.environ, {"BATZ_TIMEZONES": default_tz_str}, clear=True):
            self.bot.handle_tz(message, self.mock_client)

        self.assertIn("now", mock_subprocess_run.call_args.args[0][1])
        mock_airport_time.assert_not_called()
        self.mock_client.chat_postMessage.assert_called_once()

    @patch("main.genai.GenerativeModel")
    def test_handle_actions(self, mock_gemini):
        """Test the !actions command."""
        message = self._create_message("!actions")
        self.mock_client.conversations_replies.return_value = {
            "messages": [
                {"text": "We need to do this thing.", "user": "U1"},
            ]
        }
        self.mock_client.users_info.return_value = {
            "user": {"profile": {"display_name": "User1"}}
        }
        mock_gemini.return_value.generate_content.return_value.text = (
            "1. Do this thing."
        )
        self.bot.team_url = "https://fake.slack.com/"

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "fake-key"},
            clear=True,
        ):
            self.bot.handle_actions(message, self.mock_client)

        self.mock_client.chat_postMessage.assert_called_once()
        args, kwargs = self.mock_client.chat_postMessage.call_args
        self.assertEqual(kwargs["channel"], "C12345")
        self.assertEqual(kwargs["thread_ts"], "1628630400.000200")
        self.assertEqual(len(kwargs["blocks"]), 1)
        self.assertEqual(kwargs["blocks"][0]["type"], "section")
        self.assertIn("Do this thing", kwargs["blocks"][0]["text"]["text"])

    @patch("main.genai.GenerativeModel")
    def test_handle_genstory_gemini(self, mock_gemini):
        """Test the !genstory command with the Gemini provider."""
        message = self._create_message("!genstory")
        self.mock_client.conversations_replies.return_value = {
            "messages": [
                {"text": "This is a test", "user": "U1"},
            ]
        }
        self.mock_client.users_info.return_value = {
            "user": {"profile": {"display_name": "User1"}}
        }
        mock_gemini.return_value.generate_content.return_value.text = "Generated Story"
        self.bot.team_url = "https://fake.slack.com/"

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "fake-key"},
            clear=True,
        ):
            self.bot.handle_genstory(message, self.mock_client)

        self.mock_client.files_upload_v2.assert_called_once()
        args, kwargs = self.mock_client.files_upload_v2.call_args
        self.assertEqual(kwargs["content"], "Generated Story")

    @patch("main.requests.post")
    def test_handle_genstory_openai(self, mock_post):
        """Test the !genstory command with the OpenAI provider."""
        message = self._create_message("!genstory")
        self.mock_client.conversations_replies.return_value = {
            "messages": [
                {"text": "This is a test", "user": "U1"},
            ]
        }
        self.mock_client.users_info.return_value = {
            "user": {"profile": {"display_name": "User1"}}
        }
        mock_post.return_value.json.return_value = {
            "choices": [{"message": {"content": "Generated Story"}}]
        }
        self.bot.team_url = "https://fake.slack.com/"

        with patch.dict(
            os.environ,
            {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "fake-key"},
            clear=True,
        ):
            self.bot.handle_genstory(message, self.mock_client)

        self.mock_client.files_upload_v2.assert_called_once()
        args, kwargs = self.mock_client.files_upload_v2.call_args
        self.assertEqual(kwargs["content"], "Generated Story")

    def test_handle_copy_thread(self):
        """Test the !copyt command."""
        message = self._create_message("!copyt")
        self.mock_client.conversations_replies.return_value = {
            "messages": [
                {"text": "Hello", "user": "U1"},
                {"text": "World", "user": "U2"},
            ]
        }

        def users_info_side_effect(user):
            if user == "U1":
                return {"user": {"profile": {"display_name": "User1"}}}
            if user == "U2":
                return {"user": {"profile": {"display_name": "User2"}}}
            return {"user": {"profile": {"display_name": "Unknown"}}}

        self.mock_client.users_info.side_effect = users_info_side_effect
        self.bot.team_url = "https://fake.slack.com/"

        self.bot.handle_copy_thread(message, self.mock_client)

        self.mock_client.files_upload_v2.assert_called_once()
        args, kwargs = self.mock_client.files_upload_v2.call_args
        self.assertIn(
            "Source Thread: https://fake.slack.com/archives/C12345/p1628630400000200",
            kwargs["content"],
        )
        self.assertIn("User1: Hello", kwargs["content"])
        self.assertIn("User2: World", kwargs["content"])


if __name__ == "__main__":
    unittest.main()
