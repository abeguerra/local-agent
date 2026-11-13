---
name: notes-answer
description: Answer a question from the local notes and save the answer to a file. Use when the user wants an answer from their notes kept, saved, or written down.
---
# Answer from the notes and save it

This is the same job as the `/notes` workflow in agent/workflows.py,
written as instructions instead of code.

1. Call `search_notes` with the user's question as the query.
2. Read the snippets. If none of them actually answer the question, tell
   the user the notes don't cover it and stop. Do not save anything.
3. Write a short answer using only what the snippets say. Name the note
   file (the `source`) each fact came from.
4. Call `get_current_time` and take the date (YYYY-MM-DD) from the start
   of its `iso` value.
5. Call `write_file` with:
   - path: `answers/<date>-<slug>.md`, where the slug is the question in
     lowercase with spaces replaced by hyphens, at most 40 characters
   - content, in exactly this format:

     ```
     # <the question>

     <your answer>

     ## Sources

     - <source file>
     ```
6. Tell the user the answer and the path you saved it to.
