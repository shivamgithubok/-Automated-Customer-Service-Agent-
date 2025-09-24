class FormatPrompt:
    def generate_format_prompt(projects, formula=None):
        """
        Generates a structured prompt for Gemini to format the data.

        Args:
            projects (list): List of project names (strings).
            formula (str, optional): Formula to be included in the prompt. Default is None.

        Returns:
            str: Formatted prompt for Gemini.
        """

        # Start with the general instruction
        prompt = """
        ### Instruction for Formatting the Data:
        Please format the following information according to the specifications below. The data is related to various scientific projects and their calculations. The project names are:
        """

        # Add the list of project names
        prompt += "\n".join([f'"{project}"' for project in projects]) + "\n\n"

        # 1. Numbered List Format
        prompt += """
        ### 1. Numbered List Format
        Provide the project names in a clean, **numbered list**. Each project should be preceded by its respective number. For example:
        Example:
        1) Project Alpha
        2) Project Beta
        3) Project Gamma
        """

        # 2. Table Format (Markdown Table)
        prompt += """
        ### 2. Table Format (Markdown Table)
        Provide the project names and their descriptions in a **Markdown table** format. Each project name should be in one column, and the corresponding description (or any other data you wish to include) should be in the next column. Format it like this:
        Example (Markdown table):
        ```markdown
        | Project Name   | Description                         |
        |----------------|-------------------------------------|
        | Project Alpha  | AI-based research project          |
        | Project Beta   | Natural language processing study  |
        | Project Gamma  | Machine learning research          |
        | Project Delta  | Quantum computing exploration      |
        | Project Epsilon| Cybersecurity and cryptography     |
        | Project Zeta   | Data analytics and big data       |
        ```
        """
        prompt += """
        ### 3. Here is a reference table for different formats:
        | Format Name         | Format Type         | Example Usage                                                               | Output Example                                                       | Use Case                                                                                                                    |             |      |                  |                                     |      |               |                           |   |   |                                                                                   |
        | ------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---- | ---------------- | ----------------------------------- | ---- | ------------- | ------------------------- | - | - | --------------------------------------------------------------------------------- |
        | **Numbered List**   | Ordered list of items/projects                                              | 1) Project Alpha <br> 2) Project Beta <br> 3) Project Gamma          | When the order matters (e.g., ranking projects, listing steps in a process)                                                              |             |      |                  |                                     |      |               |                           |   |   |                                                                                   |
        | **Bullet Points**   | Unordered list of items/projects                                            | - Project Alpha <br> - Project Beta <br> - Project Gamma             | When the order doesn't matter, just showing a collection of items                                                                        |             |      |                  |                                     |      |               |                           |   |   |                                                                                   |
        | **Markdown Table**  | Structured data with two pieces of information (e.g., names & descriptions) |                                                                      | Project Name                                                                                                                             | Description | <br> | ---------------- | ----------------------------------- | <br> | Project Alpha | AI-based research project |   |   | When you need to display structured data like names with descriptions or statuses |
        | **Formula (LaTeX)** | Mathematical formula with explanation                                       | To calculate the energy of an object, we use the formula: $E = mc^2$ | When dealing with scientific, engineering, or any complex calculations that need to be formatted clearly and precisely for understanding |             |      |                  |                                     |      |               |                           |   |   |                                                                                   |
        """

        # 3. Formula Formatting (Mathematical or Other)
        if formula:
            prompt += f"""
        ### 3. Formula Formatting (Mathematical or Other)
        If any formula is required, format it using **LaTeX** or mathematical formatting. Ensure proper syntax for equations. For example, if we are dealing with energy calculations, format the formula like this:
        Example (LaTeX for energy formula):
        ```latex
        \\text{{Formula: }} \\quad {formula}
        ```
        Where:
        - \\( E \\) is the energy in joules,
        - \\( m \\) is the mass in kilograms,
        - \\( c \\) is the speed of light in meters per second.
            """

        return prompt