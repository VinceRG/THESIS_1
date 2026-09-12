"""Gemini-backed orchestration for the admin-only analytics chatbot.

This module has no Flask/SQLAlchemy dependency. It is handed a `tool_registry`
(name -> callable) built inside app.py's create_app(), because the underlying
data functions (get_dashboard_summary, role_lookup_for_diagnosis, etc.) are
closures over the Flask app/db and cannot be imported here directly. Every
registered tool is expected to be read-only and to return a JSON-serializable
dict; this module never touches the database itself.

Talks to Google's Gemini API via the `google-genai` SDK, using manually
declared function tools (not the SDK's automatic-function-calling helper) so
the carefully-tuned tool descriptions in TOOL_SCHEMAS -- which disambiguate
easily-confused tools like tool_predictions vs. tool_future_forecast -- are
preserved exactly as written instead of being re-derived from Python
docstrings.
"""

import json

from google import genai
from google.genai import types

MAX_TOOL_ITERATIONS = 4  # each round is a network round trip plus tool execution
REQUEST_TIMEOUT_SECONDS = 60

SYSTEM_PROMPT_TEMPLATE = """You are DeteK, an internal analytics assistant for Accudetek clinic administrators, embedded in the admin dashboard as a chat widget. If asked your name, say DeteK.

Current user: {role_label}, currently viewing: {branch_label}.
You are only ever used by superadmin/main_admin operators. There is no lower-privilege caller to accommodate, and this assistant is not reachable by patients or branch staff.

Rules you must always follow:
1. Answer data questions exclusively by calling the provided tools. Never assume, recall, or invent record counts, diagnoses, dates, or statistics that a tool did not return. If no tool covers the question, say so plainly rather than guessing.
2. Only pass a parameter if the question actually specifies it. Leave optional parameters (like branch_name) out entirely rather than guessing a value -- an omitted parameter is always safer than a made-up one.
3. Every recommendation (staff role, resource need, prediction) must be honest about its evidence: never present a recommendation as certain if its source is 'fallback' or `low_evidence` is true -- say plainly that the evidence is limited in that case. Convey this in plain, conversational language (e.g. "this is based on limited historical data" or "there's a strong track record for this"); never surface the tool's raw field names or jargon (`source` tier, `confidence` score, `support_count`, "historical/classifier/fallback") to the user.
4. If a tool result has zero or very few matching records, tell the user there isn't enough data for a reliable answer instead of extrapolating. Never fabricate missing records or dates.
5. The system tracks only a fixed room count constant; there is no per-room scheduling, occupancy, or booking data. If asked about room availability/scheduling beyond the total count, say plainly that this level of detail is not tracked.
6. Never reveal individual patient names, contact numbers, addresses, or any patient-identifying detail. The tools you can call never expose these, and you must never claim to know them or offer to "look them up" another way. Never reveal passwords, password hashes, API keys, environment variables, or any internal security/config detail.
7. You cannot create, edit, delete, or modify any record, user, appointment, or setting. If asked to perform an action, explain that you can only answer and analyze, and direct the user to the relevant admin page to make the change themselves.
8. The conversation history provided includes prior questions, your prior answers, and which tools backed them. Short follow-ups like "why?" refer to the most recent exchange -- resolve them using that context without asking the user to repeat themselves.
9. Be concise: lead with the key number or answer, then the supporting evidence. Avoid filler. Call at most one or two tools before answering unless truly necessary. If you already know you need more than one tool to answer, call all of them together in the same turn rather than one at a time across separate turns -- each turn is a full network round trip, so batching is significantly faster.

Whenever your answer combines a case-volume forecast with a staffing recommendation, structure it exactly like this so a clinic manager can act on it immediately:

1. The Bottom Line -- exactly one or two sentences stating the total expected cases and the exact hiring action needed. Example: "Expect around 176 total cases this December. To meet this demand, you need to add 1 Internal Medicine Physician."
2. Forecast Data -- a two-column Markdown table with headers "Condition" and "Predicted Cases" listing the top predicted cases. Do not use a bulleted list for this. Follow the table with one brief, plain-English sentence on how confident this forecast is.
3. Actionable Staffing Plan -- a bulleted list covering: the specific role that needs staff and how many more are needed (e.g. "Internal Medicine Physicians: +1 needed"); a brief plain-English reason for that gap woven directly into this same bullet (never a separate "Supporting Evidence" section); and a short confirmation that all other roles are fully staffed (or a note on any other gaps).

In these forecast-plus-staffing answers specifically: never use backend/data-science terms or AI jargon -- e.g. never say "recursive multi-step forecast," "evidence source tier," "historical mappings," "confidence score," or "low evidence." Say things the way a clinic manager would say them. Keep every sentence short and conversational.
"""


def build_system_prompt(user_context):
    return SYSTEM_PROMPT_TEMPLATE.format(
        role_label=user_context.get('role_label', 'superadmin'),
        branch_label=user_context.get('branch_label', 'All Branches'),
    )


TOOL_SCHEMAS = [
    {
        'name': 'tool_dashboard_summary',
        'description': 'Overall dashboard KPIs: total consultations, staff counts, top diagnosis, predicted next-month case load, resource readiness/capacity status, and (when viewing all branches) a per-branch breakdown.',
        'parameters': {
            'type': 'object',
            'properties': {
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to. Omit to use the branch the admin currently has selected.'},
            },
        },
    },
    {
        'name': 'tool_predictions',
        'description': 'Forecast for ONLY the single immediate next calendar month (never a month you name yourself): predicted case counts per diagnosis with low/high prediction interval and trend, plus demographic breakdown. Takes ONLY branch_name -- it has no month, year, or date parameters, and cannot be pointed at a specific month. If the user names a specific future month/year (e.g. "October 2026", "next December"), do NOT use this tool -- use tool_future_forecast instead.',
        'parameters': {
            'type': 'object',
            'properties': {
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_diagnosis_trends',
        'description': 'Trend and historical volume for a specific diagnosis (or, if omitted, the full forecasted-diagnosis and top-diagnosis lists). Use for "is X increasing", "most common diagnosis", "how many X cases".',
        'parameters': {
            'type': 'object',
            'properties': {
                'diagnosis': {'type': 'string', 'description': 'Diagnosis label as it appears in consultation records. Omit to get the overall top-diagnosis list.'},
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_staff_recommendation',
        'description': 'Evidence behind the staff-role recommendation for a diagnosis: which role(s), the evidence source tier (historical physician mapping / text-similarity classifier / static keyword fallback), confidence, and historical support count. Use this whenever the user asks which staff role handles a diagnosis, or asks "why" about a staffing/role recommendation.',
        'parameters': {
            'type': 'object',
            'properties': {
                'diagnosis': {'type': 'string', 'description': 'Diagnosis label to look up the recommended staff role for.'},
            },
            'required': ['diagnosis'],
        },
    },
    {
        'name': 'tool_department_demand',
        'description': 'Forecasted staff-role/department demand: which role has the highest predicted demand, monthly per-role demand vs. capacity, and how many diagnosis-to-role mappings were historical vs. classifier-matched vs. keyword-fallback (evidence-quality signal for staff planning).',
        'parameters': {
            'type': 'object',
            'properties': {
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_resource_capacity',
        'description': 'Day-by-day expected patient volume and staffing status (Sufficient/Monitor/Needs Staff) for the current week and next week, plus the facility room count. Use for "what should we expect today/this week", "do we have enough staff", "room capacity".',
        'parameters': {
            'type': 'object',
            'properties': {
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_query_consultations',
        'description': 'Search/aggregate consultation records by diagnosis, department, and/or date range. Returns match counts and per-diagnosis/department breakdowns -- never patient names or contact info. Use for "how many X consultations", "most common diagnosis last month".',
        'parameters': {
            'type': 'object',
            'properties': {
                'diagnosis': {'type': 'string', 'description': 'Filter to this diagnosis (partial match allowed).'},
                'department': {'type': 'string', 'description': 'Filter to this department (partial match allowed).'},
                'date_from': {'type': 'string', 'description': 'ISO date (YYYY-MM-DD), inclusive lower bound.'},
                'date_to': {'type': 'string', 'description': 'ISO date (YYYY-MM-DD), inclusive upper bound.'},
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_historical_analysis',
        'description': 'Grouped historical consultation counts over a date range (by diagnosis, department, age_group, gender, or month). Use for seasonal-pattern or historical-comparison questions.',
        'parameters': {
            'type': 'object',
            'properties': {
                'date_from': {'type': 'string', 'description': 'ISO date (YYYY-MM-DD), inclusive lower bound.'},
                'date_to': {'type': 'string', 'description': 'ISO date (YYYY-MM-DD), inclusive upper bound.'},
                'group_by': {'type': 'string', 'description': "One of: diagnosis, department, age_group, gender, month. Defaults to diagnosis."},
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
            'required': ['date_from', 'date_to'],
        },
    },
    {
        'name': 'tool_backtest_forecast',
        'description': "Compares what the model would have predicted for a given diagnosis and past month (trained only on data before that month) against the actual recorded count for that month. Use for \"how accurate was last month's forecast\" / predicted-vs-actual questions.",
        'parameters': {
            'type': 'object',
            'properties': {
                'diagnosis': {'type': 'string', 'description': 'Diagnosis label to backtest.'},
                'target_month': {'type': 'integer', 'description': 'Month number (1-12) to backtest.'},
                'target_year': {'type': 'integer', 'description': 'Four-digit year to backtest.'},
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
            'required': ['diagnosis', 'target_month', 'target_year'],
        },
    },
    {
        'name': 'tool_future_forecast',
        'description': 'Forecasts top diagnoses for a month the user names explicitly by month and year (e.g. "October 2026", "next December", "top cases in March"). Use this any time a specific month/year is named in the question, even if it happens to be next month by name -- only use tool_predictions when the user vaguely says "next month" with no month named. Always relay the returned `confidence` and `months_out_from_next_forecastable_month` fields honestly; this gets less reliable the further out the target month is, and the tool refuses months too far out on its own.',
        'parameters': {
            'type': 'object',
            'properties': {
                'target_month': {'type': 'integer', 'description': 'Month number (1-12) to forecast.'},
                'target_year': {'type': 'integer', 'description': 'Four-digit year to forecast.'},
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
            'required': ['target_month', 'target_year'],
        },
    },
    {
        'name': 'tool_model_metrics',
        'description': 'Forecasting model quality: R-squared, MAE, MSE, RMSE, cross-validation scores, and a plain-language verdict. Use for "how good/accurate is the model" questions.',
        'parameters': {
            'type': 'object',
            'properties': {
                'branch_name': {'type': 'string', 'description': 'Optional branch name to scope the answer to.'},
            },
        },
    },
    {
        'name': 'tool_generate_report',
        'description': "Generate one of the existing report payloads. report_key must be one of: monthly-consultation, quarterly, prediction, resource-recommendation, attribution-gaps.",
        'parameters': {
            'type': 'object',
            'properties': {
                'report_key': {'type': 'string', 'description': 'One of: monthly-consultation, quarterly, prediction, resource-recommendation, attribution-gaps.'},
            },
            'required': ['report_key'],
        },
    },
    {
        'name': 'tool_chatbot_help',
        'description': 'Describes what this assistant can and cannot do. Use for "what can you do" / "how do I use this" questions.',
        'parameters': {'type': 'object', 'properties': {}},
    },
]


def _build_gemini_tools():
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=schema['name'],
            description=schema['description'],
            parameters=schema['parameters'],
        )
        for schema in TOOL_SCHEMAS
    ])


class GeminiError(Exception):
    """Raised when the Gemini API can't be reached or returns an error."""


def _call_gemini(client, model_name, contents, system_prompt, tools):
    """Thin, mockable seam around the Gemini API."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=[tools],
    )
    try:
        return client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - the SDK raises its own exception hierarchy
        raise GeminiError(f'Could not get a response from Gemini ({model_name}): {exc}') from exc


def run_chatbot_turn(question, tool_registry, history, user_context, api_key, model_name, on_tool_call=None):
    """Runs one chatbot turn to completion (including any tool-call round trips).

    history: list of {'q': str, 'a': str} for prior turns in this conversation
    (most-recent-last), used so short follow-ups like "why?" resolve correctly.

    on_tool_call: optional callback invoked with the tool name right before
    each tool actually runs, so a caller can surface live status ("Checking
    staffing demand...") to the user while a slow tool is executing.

    Returns (answer_text, tools_used) where tools_used is a list of
    {'tool': name, 'args': {...}} for every tool call actually made.
    """
    if not api_key:
        raise GeminiError('GEMINI_API_KEY is not configured.')

    client = genai.Client(api_key=api_key)
    tools = _build_gemini_tools()
    system_prompt = build_system_prompt(user_context)

    contents = []
    for turn in history:
        contents.append(types.Content(role='user', parts=[types.Part(text=turn['q'])]))
        contents.append(types.Content(role='model', parts=[types.Part(text=turn['a'])]))
    contents.append(types.Content(role='user', parts=[types.Part(text=question)]))

    tools_used = []
    for _ in range(MAX_TOOL_ITERATIONS):
        response = _call_gemini(client, model_name, contents, system_prompt, tools)
        candidates = response.candidates or []
        if not candidates or not candidates[0].content:
            return (response.text or '').strip(), tools_used

        model_content = candidates[0].content
        contents.append(model_content)

        function_calls = [part.function_call for part in (model_content.parts or []) if part.function_call]
        if not function_calls:
            return (response.text or '').strip(), tools_used

        response_parts = []
        for call in function_calls:
            name = call.name
            # Small models sometimes pass null instead of omitting an
            # optional argument -- treat null the same as "not provided".
            args = {key: value for key, value in dict(call.args or {}).items() if value is not None}
            tool_fn = tool_registry.get(name)
            if tool_fn is None:
                result = {'error': f'Unknown tool: {name}'}
            else:
                if on_tool_call:
                    on_tool_call(name)
                try:
                    result = tool_fn(**args)
                except Exception as exc:  # noqa: BLE001 - surfaced to the model as data, not a crash
                    result = {'error': str(exc)}
            tools_used.append({'tool': name, 'args': args})
            response_parts.append(types.Part.from_function_response(
                name=name,
                response={'result': json.loads(json.dumps(result, default=str))},
            ))
        contents.append(types.Content(role='user', parts=response_parts))

    return (
        "I wasn't able to finish that after several tool calls. Please rephrase the question or ask something narrower.",
        tools_used,
    )
