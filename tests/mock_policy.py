from __future__ import annotations
from TableAgent.QA.actions.base_action import BaseCodeGenerationAction, CodeGenerationRequest, CodeGenerationResult

class MockActionPolicy(BaseCodeGenerationAction):
    """
    A deterministic policy for mock code generation based on question keywords and subtask layers.
    Also supports simulating error injection to test ReAct's self-repair functionality.
    """
    def __init__(self, simulate_error: bool = False):
        self.simulate_error = simulate_error

    def run(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """
        Returns generated code and description based on question and layer.
        """
        question = request.question
        layer = request.layer
        round_num = request.round_num
        subtask = request.subtask
        q_lower = question.lower()
        table_id = "table1"
        if subtask and hasattr(subtask, "metadata") and isinstance(subtask.metadata, dict):
            table_id = subtask.metadata.get("table_id", "table1")

        # Simulate error on round 1 if requested
        if self.simulate_error and round_num == 1:
            code = (
                "# Simulating a syntax/runtime error for ReAct repair testing\n"
                "# Calling a non-existent method to trigger AttributeError\n"
                "scores = operators.read_range_nonexistent('G3:G22')\n"
            )
            desc = "Attempts to read scores using an invalid operator method."
            return CodeGenerationResult(code=code, description=desc)

        # Normal/Correct logic
        if layer == "inspect":
            if "score" in q_lower:
                code = (
                    f"first_hdr = operators.get_header('{table_id}', 'first_name')\n"
                    f"middle_hdr = operators.get_header('{table_id}', 'middle_name')\n"
                    f"last_hdr = operators.get_header('{table_id}', 'last_name')\n"
                    f"score_hdr = operators.get_header('{table_id}', 'score')\n"
                    "\n"
                    "firsts = operators.read_range_flat(first_hdr.data_range)\n"
                    "middles = operators.read_range_flat(middle_hdr.data_range)\n"
                    "lasts = operators.read_range_flat(last_hdr.data_range)\n"
                    "scores = operators.read_range_flat(score_hdr.data_range)\n"
                    "\n"
                    "people = []\n"
                    "for f, m, l, s in zip(firsts, middles, lasts, scores):\n"
                    "    people.append({\n"
                    "        'name': f'{f} {m} {l}',\n"
                    "        'score': s\n"
                    "    })\n"
                    "print(f'Loaded {len(people)} people records with scores.')\n"
                )
                desc = f"Extract first name, middle name, last name, and scores into a unified people list from {table_id}."
            elif "birth" in q_lower or "date" in q_lower or "born" in q_lower:
                code = (
                    f"first_hdr = operators.get_header('{table_id}', 'first_name')\n"
                    f"middle_hdr = operators.get_header('{table_id}', 'middle_name')\n"
                    f"last_hdr = operators.get_header('{table_id}', 'last_name')\n"
                    f"birth_hdr = operators.get_header('{table_id}', 'birth_date')\n"
                    "\n"
                    "firsts = operators.read_range_flat(first_hdr.data_range)\n"
                    "middles = operators.read_range_flat(middle_hdr.data_range)\n"
                    "lasts = operators.read_range_flat(last_hdr.data_range)\n"
                    "births = operators.read_range_flat(birth_hdr.data_range)\n"
                    "\n"
                    "people = []\n"
                    "for f, m, l, b in zip(firsts, middles, lasts, births):\n"
                    "    people.append({\n"
                    "        'first': f,\n"
                    "        'middle': m,\n"
                    "        'last': l,\n"
                    "        'birth': b\n"
                    "    })\n"
                    "print(f'Loaded {len(people)} people records with birth dates.')\n"
                )
                desc = f"Extract names and birth dates into a people list from {table_id}."
            else:
                # Default inspection
                code = (
                    f"headers = operators.list_headers('{table_id}')\n"
                    f"print(f'Found {{len(headers)}} headers in {table_id}.')\n"
                )
                desc = f"List all headers in {table_id}."
        else:
            # Synthesis layer
            if "average score" in q_lower or "mean score" in q_lower:
                code = (
                    "scores_list = [p['score'] for p in people if p['score'] is not None]\n"
                    "avg_score = sum(scores_list) / len(scores_list) if scores_list else 0\n"
                    "final_answer = f'{avg_score:.2f}'\n"
                    "print(f'Average score is: {final_answer}')\n"
                )
                desc = "Aggregate the scores of all people and calculate the average."
            elif "highest score" in q_lower or "max score" in q_lower:
                code = (
                    "best_person = max(people, key=lambda p: p['score'] if p['score'] is not None else -1)\n"
                    "final_answer = best_person['name']\n"
                    "print(f'Highest score belongs to: {final_answer}')\n"
                )
                desc = "Find the person with the highest score."
            elif "birth date of an gia pham" in q_lower:
                code = (
                    "target = None\n"
                    "for p in people:\n"
                    "    if p['first'] == 'An' and p['middle'] == 'Gia' and p['last'] == 'Pham':\n"
                    "        target = p\n"
                    "        break\n"
                    "if target:\n"
                    "    # Format date to YYYY-MM-DD\n"
                    "    date_val = target['birth']\n"
                    "    if hasattr(date_val, 'strftime'):\n"
                    "        final_answer = date_val.strftime('%Y-%m-%d')\n"
                    "    else:\n"
                    "        final_answer = str(date_val)\n"
                    "else:\n"
                    "    final_answer = 'Not Found'\n"
                    "print(f'Birth date of An Gia Pham: {final_answer}')\n"
                )
                desc = "Filter extracted people for An Gia Pham and retrieve their formatted birth date."
            else:
                code = (
                    "final_answer = 'Success'\n"
                    "print('Synthesis complete.')\n"
                )
                desc = "Return a default successful synthesis result."

        return CodeGenerationResult(code=code, description=desc)

