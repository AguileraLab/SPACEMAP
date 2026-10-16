"""Conversation proposals and execution receipts; the command bus remains the executor."""

import hashlib
import json
import re


def action_envelope(value):
    """Normalize provider JSON spellings; argument validation remains mandatory."""
    from .schemas import COMMANDS

    if not isinstance(value, dict):
        return None
    if isinstance(value.get("action"), dict) or "plan" in value:
        return value
    if isinstance(value.get("action"), str):
        return {"say":value.get("say", ""), "action":{"name":value["action"], "args":value.get("args", {})}}
    if isinstance(value.get("name"), str) and "args" in value:
        return {"say":value.get("say", ""), "action":{"name":value["name"], "args":value["args"]}}
    names = [key for key in value if key in COMMANDS]
    if len(names) == 1 and set(value) <= {names[0], "say"} and isinstance(value[names[0]], dict):
        return {"say":value.get("say", ""), "action":{"name":names[0], "args":value[names[0]]}}
    return None


def cluster_args(args, bound_job):
    """Bind names to the reviewed run, never search for a similarly named run.

    An omitted ID or a dropped digit in a 32-hex ID can use the known context.
    A different valid ID, invented ID, or invalid argument shape is rejected.
    """
    if not bound_job or not isinstance(args, dict):
        return args
    supplied = args.get("job_id")
    if supplied and supplied != bound_job:
        remaining = iter(bound_job)
        dropped = (isinstance(supplied, str) and re.fullmatch(r"[0-9a-f]{30,31}", supplied)
                   and re.fullmatch(r"[0-9a-f]{32}", bound_job) and all(char in remaining for char in supplied))
        if not dropped:
            raise ValueError("The returned names refer to a different run. Review the selected run again.")
    return {**args, "job_id":bound_job}


def json_object(text):
    """Read an embedded JSON object, including mismatched final container closers.

    Repair only closing brackets after complete values; never invent a name, value,
    quote or comma. An incomplete string/number is not a usable action.
    """
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        fragment = text[start:]
        try:
            value, _ = decoder.raw_decode(fragment)
        except ValueError:
            stack, parts, quoted, escaped = [], [], False, False
            for char in fragment:
                if quoted:
                    parts.append(char)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        quoted = False
                    continue
                if char == '"':
                    quoted = True
                elif char in "{[":
                    stack.append("}" if char == "{" else "]")
                elif char in "}]":
                    if char not in stack:
                        break
                    while stack[-1] != char:
                        parts.append(stack.pop())
                    stack.pop()
                parts.append(char)
                if not stack:
                    break
            # Some providers stop normally after the last complete object but
            # omit the outer envelope's closer. Complete only containers after
            # an already closed object/array; never complete strings or values.
            if not quoted and stack and fragment.rstrip().endswith(('}', ']')):
                parts.extend(reversed(stack))
                stack.clear()
            if quoted or stack:
                continue
            try:
                value = json.loads("".join(parts))
            except ValueError:
                continue
        envelope = action_envelope(value)
        if envelope is not None:
            return envelope
    return None


def approval(text):
    return bool(re.fullmatch(r"(?:(?:ok(?:ay)?|yes|sure|please)[,.!]?\s*)*(?:do (?:it|that)|apply (?:it|that|those(?: names)?)|go ahead|save (?:them|those names)|proceed|(?:name|label) (?:the|these|those) clusters? then)[.!]?", text.strip(), re.I))


def requests_action(text):
    """A request to edit/run, rather than a question or a request for suggestions."""
    if re.search(r"\b(?:do not|don['’]t|without|before|not yet|only suggest|for review only)\b", text, re.I):
        return False
    if re.match(r"\s*(?:how|why|what|which|explain|suggest|recommend|would it|what if)\b", text, re.I):
        return False
    return bool(re.search(r"(?:^|[.!?\n]\s*|\b(?:can|could|would|will) you (?:please )?|\bI (?:want|need)(?: you)? to )"
                          r"(?:please\s+)?(?:label|name|assign|apply|save|write|rename|change|update|clear|remove|undo|run|start|set|correct|segment|detect|export)\b", text.strip(), re.I))


def authorizes(text, action):
    """Only execute changes of the kind the user requested, not unrelated model actions."""
    from .schemas import COMMANDS
    name = action["name"]
    if not COMMANDS[name].changes_project:
        return True
    if not requests_action(text):
        return False
    targets = {
        "name_clusters":r"cluster|phenotype|names|labels",
        "assign_selection":r"selected|selection|cells",
        "change_settings":r"settings|resolution|neighbours|neighbors|markers|normalis|normaliz",
        "set_threshold":r"threshold", "save_thresholds":r"threshold",
    }
    if name.startswith("run_"):
        targets[name] = re.escape(name[4:]).replace("_", r"[ _-]")
    return bool(name in targets and re.search(targets[name], text, re.I))


def proposal(store, conversation, message):
    found = next((item for item in store.get_messages(conversation, limit=300) if item["id"] == message), None)
    if not found or found["role"] != "assistant":
        raise ValueError("That assistant action is no longer available. Ask again.")
    return found


def receipt(store, conversation, message, index=0):
    messages = store.get_messages(conversation, limit=300)
    for item in messages:
        if item["id"] == message:
            message = (item.get("metadata") or {}).get("proposal_source") or message
    for item in reversed(messages):
        meta = item.get("metadata") or {}
        if meta.get("source") == "studio_execution" and meta.get("proposal") == message and meta.get("index") == index:
            return meta
    return None


def execution_status(meta, root):
    if not meta:
        return None
    found = dict(meta)
    if found.get("job_id"):
        from . import jobs
        try:
            job = jobs.status(root, found["job_id"])
            found.update(status=job["status"], message=job.get("message", ""), column=(job.get("result") or {}).get("column"))
        except (OSError, ValueError, KeyError):
            found.update(status="unknown", message="Check this job in Jobs; its status could not be read.")
    return found


def latest_proposal(store, conversation, root, selected_job=None):
    from .chat import read_reply
    for item in reversed(store.get_messages(conversation, limit=100)):
        if item["role"] != "assistant":
            continue
        meta = item.get("metadata") or {}
        if meta.get("project") and meta["project"] != str(root):
            continue
        action, plan = meta.get("action"), meta.get("plan")
        if not action and not plan:
            bound = (meta.get("cluster_review") or {}).get("job_id") or meta.get("review_job_id") or selected_job
            _, action, plan = read_reply(item["content"], cluster_job_id=bound)
        if selected_job and any(a["name"] == "name_clusters" and a["args"]["job_id"] != selected_job for a in ([action] if action else plan or [])):
            continue
        if action or plan:
            statuses = [execution_status(receipt(store, conversation, item["id"], index), root)
                        for index in range(len(plan or [action]))]
            status = next((s for s in statuses if s and s["status"] in {"starting", "running", "stopping"}), None)
            if not status and all(s and s["status"] == "done" for s in statuses):
                status = statuses[-1]
            return item, action, plan, status
    return None


def prepare_command(channel, body, store):
    """Attach an idempotent receipt to a reviewed/explicitly requested chat command."""
    from spacemap_gui.viewer import registry
    context = body.get("assistant")
    if not context:
        return body, None
    conversation, message = context.get("conversation"), context.get("message")
    item = proposal(store, conversation, message)
    meta = item.get("metadata") or {}
    root = (registry.session_project(channel) or {}).get("root")
    if meta.get("project") != str(root):
        raise ValueError("This action belongs to a different project. Ask again in the current project.")
    from .chat import read_reply
    _, parsed_action, parsed_plan = read_reply(item["content"])
    steps = meta.get("plan") or ([meta.get("action") or parsed_action] if (meta.get("action") or parsed_action) else parsed_plan) or []
    index = context.get("index", 0)
    if not isinstance(index, int) or not 0 <= index < len(steps) or steps[index]["name"] != body.get("name"):
        raise ValueError("This command does not match the assistant action.")
    origin = meta.get("proposal_source") or message
    request_id = "chat_" + hashlib.sha256(f"{conversation}:{origin}:{index}".encode()).hexdigest()[:32]
    return {**body, "request_id":request_id}, {"conversation":conversation, "proposal":origin, "index":index, "project":str(root), "args":body.get("args", {})}


def record(store, context, result):
    if not context or result.get("status") == "needs_confirmation":
        return
    job = (result.get("data") or {}).get("job") or {}
    meta = {**context, "source":"studio_execution", "command_id":result.get("command_id"),
            "name":result.get("name"), "status":job.get("status") or result["status"],
            "message":result.get("message", ""), "job_id":job.get("id")}
    previous = receipt(store, context["conversation"], context["proposal"], context["index"])
    if previous and all(previous.get(key) == value for key, value in meta.items()):
        return
    store.add_message(context["conversation"], "system", meta["message"], metadata=meta)


def direct_cluster_edit(text, page):
    """Exact cluster edits need no language model; biological inference still does."""
    job = (page or {}).get("explorer", {}).get("job_id")
    if not job or not requests_action(text):
        return None
    clean = re.sub(r"^(?:(?:please|now)\s+|(?:can|could|would|will) you (?:please )?)+", "", text.strip(), flags=re.I).rstrip("?.!")
    edit = re.fullmatch(r"(?:label|name|rename|call|change)\s+(Cluster\s+\d+)\s+(?:as|to|=)\s+(.+)", clean, re.I)
    if edit:
        return {"name":"name_clusters", "args":{"job_id":job, "names":{"Cluster " + edit[1].split()[-1]:edit[2].strip(" \"'")}}}
    clear = re.fullmatch(r"(?:clear|remove|undo)\s+(?:the\s+)?(?:name|label)s?\s+(?:of|for|from)\s+(Cluster\s+\d+)", clean, re.I)
    if clear:
        return {"name":"name_clusters", "args":{"job_id":job, "clear":["Cluster " + clear[1].split()[-1]]}}
    if re.fullmatch(r"(?:clear|remove)\s+(?:all\s+)?(?:the\s+)?cluster (?:names|labels)", clean, re.I):
        return {"name":"name_clusters", "args":{"job_id":job, "replace":True}}
    return None
