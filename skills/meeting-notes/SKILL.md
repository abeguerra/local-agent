---
name: meeting-notes
description: Turn rough meeting notes into a tidy, consistently formatted summary file. Use when the user pastes or describes notes from a meeting, call, or standup.
---
# Meeting notes

1. Call `get_current_time` and take today's date (YYYY-MM-DD) from the
   start of its `iso` value, unless the user gave the meeting's date.
2. From the user's notes, pull out:
   - who attended
   - what was decided
   - action items, each with an owner and a due date if one was given
   Use only what the user said. If something is missing, such as an
   action item with no owner, write "unassigned" rather than inventing
   one.
3. Call `write_file` with path `meetings/<date>-<topic>.md`, where the
   topic is two to four lowercase words joined by hyphens, and content
   in exactly this format:

   ```
   # <Topic> — <date>

   **Attendees:** <names, comma-separated>

   ## Decisions

   - <decision>

   ## Action items

   - [ ] <task> — <owner> (due <date or "no date">)
   ```

   Leave a section with a single "- none" line if there is nothing for it.
4. Reply with the action items and the path you saved to. Don't repeat
   the whole file.
