# ec-logic-tool

Based on our conversation history and the code we have built together, here is the current format and logic that the application understands and implements:

1. The Input Format 📝
The application expects you to enter complex logical strings into the "Dynamic Request Input" text areas (REQ - 1, REQ - 2, etc.).

Structure: It handles boolean logic using AND, OR, (, and ).

Code Pattern: It looks for specific "Codes" (format XXX_XX, e.g., LYQ_01, AJA_02).

Flexibility: It is smart enough to find these codes even if they are buried in complex text like:

HP-0000873 [ DICO... ] AND ( ... )

HP-0000873[LYQ{LYQ_01}] (It extracts LYQ_01 from inside the braces).

Standard text: LYQ_01 AND AJA_02

2. The Processing Logic ⚙️
When you click "Process Information", the tool performs three distinct steps:

Step A: Cleaned Logic Output

It strips away all "garbage" text (like "HP-0000...", "DICO...", descriptions).

It retains only the AND/OR operators, parentheses (), and the valid Codes.

Example: (LYQ_01 AND AJA_02) OR CZB_01

Step B: Generated Combinations

It flattens the logic into unique paths (Disjunctive Normal Form).

Rules Applied:

Separator: AND becomes | (Pipe).

Lines: OR becomes a New Line.

Spacing: No spaces allowed.

Example: AJA_02|LYQ_01 (new line) CZB_01

Step C: Validation & Filtering

It checks every generated combination against your uploaded Excel data.

It looks at the columns: Used from DPEO and Used from CDPO.

Logic:

If all codes in a line have Y for DPEO → marked as --> DPEO.

If all codes have Y for CDPO → marked as --> CDPO.

If neither → marked as --> NOT APPLICABLE to NEA.

3. The Final Output Format 📊
The results are displayed in four specific sections for each Request:

Cleaned Logic Output: The sanitized logic string.

Generated Combinations: The raw Code|Code|Code list.

Combination Validation:

Displays all combinations with their status (--> DPEO, --> CDPO, etc.).

Color Coded: Green for valid lines, Red for "NOT APPLICABLE" lines.

Visual: Badges/Pills style.

Final Combination:

This is a Strictly Filtered list.

Included: Lines valid for CDPO or DPEO & CDPO.

Excluded: Lines that are "NOT APPLICABLE" or purely "DPEO" (as per your last request).

Feature: Includes a Download button.

4. Visual Style 🎨
Theme: Modern "Soft UI" with a Teal/Green color scheme (#008080).

Components: Card-based layout with soft shadows, interactive Toast notifications, and summary metrics at the top.

Tables: Interactive tables where Y/N values are converted to Checkboxes.
