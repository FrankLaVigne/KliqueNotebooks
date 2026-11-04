"""Offline tests: python -m unittest discover -s 05_VSCode -p test_agent_lab.py -v"""
import json
import unittest
from agent_lab import MockClient, calculator, demo_client, mock_reply, run_agent, run_tool


class Rejection(Exception):
    def __init__(self, kind, status_code=429):
        self.kind, self.status_code = kind, status_code
        super().__init__("Synthetic rejection, no gateway contacted")


def native_call(arguments='{"expression":"17 * 23"}', name="calculator"):
    return {"id": "call_lab_1", "type": "function",
            "function": {"name": name, "arguments": arguments}}


class AgentTests(unittest.TestCase):
    def test_json_valid_tools_and_history(self):
        client = demo_client()
        run = run_agent(client)
        self.assertEqual(run["status"], "done")
        self.assertEqual(run["answer"], "$60.00 for 3 hours")
        self.assertEqual(run["tokens"], {"prompt": 30, "completion": 15})
        self.assertEqual(len(client.calls), 3)
        self.assertIn('60.0', client.calls[2]["messages"][-1]["content"])
        self.assertIsNone(client.calls[0]["tools"])
        self.assertEqual(run["history"][-1]["role"], "assistant")

    def test_native_valid_tool(self):
        client = MockClient([mock_reply(None, tool_calls=[native_call()]), mock_reply("391")])
        run = run_agent(client, mode="native")
        self.assertEqual(run["status"], "done")
        tool_message = client.calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(tool_message["tool_call_id"], "call_lab_1")
        self.assertEqual(json.loads(tool_message["content"])["result"], 391)
        self.assertEqual(client.calls[0]["tool_choice"], "auto")

    def test_malformed_json_stops_after_two_responses(self):
        client = MockClient([mock_reply("not json")] * 2)
        run = run_agent(client)
        self.assertEqual(run["status"], "parse_error")
        self.assertEqual(len(client.calls), 2)

    def test_malformed_json_can_recover(self):
        client = MockClient([mock_reply("not json"), mock_reply('{"final":"recovered"}')])
        self.assertEqual(run_agent(client)["answer"], "recovered")

    def test_native_malformed_arguments_stop(self):
        client = MockClient([mock_reply(None, tool_calls=[native_call("{")])])
        self.assertEqual(run_agent(client, mode="native")["status"], "parse_error")
        self.assertEqual(len(client.calls), 1)

    def test_unknown_tool_returns_error_and_does_not_execute(self):
        client = MockClient([mock_reply('{"tool":"shell","arguments":{"command":"echo unsafe"}}'),
                             mock_reply('{"final":"Tool unavailable"}')])
        run = run_agent(client)
        self.assertEqual(run["status"], "done")
        self.assertIn("unknown tool", client.calls[1]["messages"][-1]["content"])

    def test_bad_argument_types_are_tool_errors(self):
        self.assertIn("error", run_tool("calculator", []))
        self.assertIn("error", run_tool("calculator", {"expression": "1+1", "extra": 1}))
        self.assertIn("error", run_tool("lookup", {"key": []}))

    def test_rate_limit_exhaustion_stops_without_retry(self):
        client = MockClient([Rejection("rate_limit")])
        run = run_agent(client)
        self.assertEqual(run["status"], "stopped_by_gateway")
        self.assertIn("rate_limit", run["trace"][0]["note"])
        self.assertEqual(len(client.calls), 1)

    def test_budget_rejection_stops_without_retry(self):
        client = MockClient([Rejection("budget")])
        run = run_agent(client)
        self.assertEqual(run["status"], "stopped_by_gateway")
        self.assertEqual(len(client.calls), 1)

    def test_unknown_429_is_not_assumed_to_be_rate_limit(self):
        client = MockClient([Rejection("ambiguous_limit")])
        run = run_agent(client)
        self.assertIn("ambiguous_limit", run["trace"][0]["note"])
        self.assertEqual(len(client.calls), 1)

    def test_other_gateway_failures_do_not_change_mode_or_retry(self):
        for kind, status in [("authentication", 401), ("forbidden", 403), ("not_found", 404),
                             ("request_error", 400), ("timeout", None)]:
            with self.subTest(kind=kind):
                client = MockClient([Rejection(kind, status)])
                self.assertEqual(run_agent(client, mode="native")["status"], "stopped_by_gateway")
                self.assertEqual(len(client.calls), 1)

    def test_step_cap(self):
        client = demo_client()
        self.assertEqual(run_agent(client, max_steps=1)["status"], "max_steps")
        self.assertEqual(len(client.calls), 1)

    def test_token_cap_at_boundary_before_tool_execution(self):
        client = demo_client()
        run = run_agent(client, max_run_tokens=15)
        self.assertEqual(run["status"], "token_limit")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(run["history"][-1]["role"], "assistant")
        self.assertEqual(run["trace"][0]["tool"], "")

    def test_token_cap_overrides_final_answer(self):
        client = MockClient([mock_reply('{"final":"too expensive"}', prompt=100, completion=50)])
        run = run_agent(client, max_run_tokens=100)
        self.assertEqual(run["status"], "token_limit")
        self.assertIsNone(run["answer"])
        self.assertEqual(sum(run["tokens"].values()), 150)

    def test_missing_usage_fails_closed(self):
        client = MockClient([{"choices": [{"message": {"content": '{"final":"ok"}'}}]}])
        self.assertEqual(run_agent(client)["status"], "usage_unavailable")

    def test_missing_choices_is_reported(self):
        client = MockClient([{"usage": {"prompt_tokens": 1, "completion_tokens": 1}}])
        self.assertEqual(run_agent(client)["status"], "invalid_response")

    def test_empty_final_is_not_success(self):
        for mode, content in [("json", '{"final":""}'), ("native", "")]:
            with self.subTest(mode=mode):
                client = MockClient([mock_reply(content)])
                self.assertEqual(run_agent(client, mode=mode, max_parse_errors=1)["status"], "parse_error")

    def test_safe_calculator(self):
        self.assertEqual(calculator("8 * 2.5 * 3"), 60)
        for expression in ["__import__('os')", "2 ** 1000000", "True + 1", "1 / 0", "1e100"]:
            with self.subTest(expression=expression):
                self.assertIn("error", run_tool("calculator", {"expression": expression}))

    def test_clock_and_lookup_are_local_tools(self):
        self.assertIn("UTC", run_tool("clock", {})["result"])
        self.assertEqual(run_tool("lookup", {"key": "lab-a100"})["result"]["gpus"], 8)

    def test_bad_limits_rejected_before_requests(self):
        client = demo_client()
        with self.assertRaises(ValueError):
            run_agent(client, max_steps=0)
        self.assertEqual(client.calls, [])


if __name__ == "__main__":
    unittest.main()
