from pydantic import BaseModel


class Motivation(BaseModel):
    description: str
    problem_to_solve: str
    how_to_solve: str
    why_can_be_solved: str

    def __str__(self):
        split_line = "*" * 20 + "\n"
        return f"""
        {split_line}
        description:{self.description}
        problem_to_solve:{self.problem_to_solve}
        how_to_solve:{self.how_to_solve}
        why_can_be_solved:{self.why_can_be_solved}
        {split_line}
        """


class Conclusion(BaseModel):
    description: str
    problem_to_solve: str
    how_much_is_solved: str
    contribution: str

    def __str__(self):
        split_line = "*" * 20 + "\n"
        return f"""
        {split_line}
        description:{self.description}
        problem_to_solve:{self.problem_to_solve}
        how_much_is_solved:{self.how_much_is_solved}
        contribution:{self.contribution}
        {split_line}
        """


class BulletPoint(BaseModel):
    subject: str
    description: str
    technical_details: str
    importance_explanation: str
    importance: int

    def __str__(self):
        split_line = "*" * 20 + "\n"
        return f"""
        {split_line}
        subject:{self.subject}
        description:{self.description}
        importance_explanation:{self.importance_explanation}
        technical_details:{self.technical_details}
        importance:{self.importance}
        {split_line}
        """

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "technical_details": {"type": "string"},
                "importance_explanation": {"type": "string"},
                "importance": {"type": "integer"},
            },
            "required": [
                "subject",
                "description",
                "technical_details",
                "importance_explanation",
                "importance",
            ],
        }


class InnovationPoint(BaseModel):
    subject: str
    description: str
    technical_details: str
    why_is_innovative: str
    similar_concepts: list[str]
    importance: int

    def __str__(self):
        split_line = "*" * 20 + "\n"
        return f"""
        {split_line}
        subject:{self.subject}
        description:{self.description}
        technical_details:{self.technical_details}
        why_is_innovative:{self.why_is_innovative}
        similar_concepts:{self.similar_concepts}
        importance:{self.importance}
        {split_line}
        """

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "technical_details": {"type": "string"},
                "why_is_innovative": {"type": "string"},
                "similar_concepts": {"type": "array", "items": {"type": "string"}},
                "importance": {"type": "integer"},
            },
            "required": [
                "subject",
                "description",
                "technical_details",
                "why_is_innovative",
                "similar_concepts",
                "importance",
            ],
        }


class BaseSummary(BaseModel):
    """Base summary class with motivation and bullet points"""

    motivation: Motivation
    bullet_points: list[BulletPoint]

    def __str__(self):
        split_line = "=" * 20 + "\n"
        return f"""
        {split_line}
        motivation:{self.motivation}
        bullet_points:{self.bullet_points}
        {split_line}
        """

    @classmethod
    def get_rag_key_words(cls) -> list[str]:
        """Returns a list of key words important for RAG purposes."""
        return ["base_summary"]

    @classmethod
    def model_json_schema(cls):
        schema = super().model_json_schema()
        schema["required"] = list(schema["properties"].keys())
        return schema

    @classmethod
    def excluded_fields(cls) -> list[str]:
        """Returns a list of fields that should be excluded during question set generation."""
        return []

    @classmethod
    def prompt(cls):
        return """
I want you to act as a professional summarizer tasked with breaking down the provided input material into concise, clearly separated bullet points. Each bullet point must capture a single, distinct concept, idea, or piece of information from the material. Follow these detailed instructions to ensure clarity and thoroughness:

Role and Objective
Role: Act as an expert summarizer with a focus on clarity and organization.
Objective: Produce a list of bullet points where each point represents a standalone concept from the input material. Avoid combining multiple ideas into one point.
Steps to Follow
Analyze the Material:

Carefully read through the entire material to understand its structure, main topics, subtopics, and detailed content.
Identify distinct ideas, arguments, or pieces of information that are critical to understanding the material.
Break Down the Content:

Divide the material into sections, paragraphs, or logical groupings.
Extract key concepts or pieces of information from each section.
Focus on isolating each idea into a standalone statement for clarity.
Create Bullet Points:

Write bullet points that are concise yet complete, clearly conveying a single concept.
Avoid combining multiple ideas into one bullet point. If necessary, break down complex ideas into smaller, separate points.
Use precise and straightforward language to ensure clarity.
Ensure Separation of Concepts:

Review each bullet point to confirm it reflects a unique idea and does not overlap with other points.
Ensure that each concept is independently understandable without requiring context from other bullet points.
Prioritize Relevance:

Focus on the most important ideas or findings in the material.
Exclude minor details or redundant information unless they are critical for understanding the main concepts.
Structure the Summary:

Present the bullet points in a logical order, following the structure of the input material (e.g., section by section).
If the material has a hierarchical structure, group related bullet points under subheadings for better organization.
Example Output Format
If the material is about "The Impact of Renewable Energy on Global Energy Markets," the summary might look like this:

Additional Considerations:
Maintain neutrality and avoid inserting opinions or interpretations.
If technical terms or jargon are used in the material, retain them in the bullet points but keep the explanations straightforward.
For lengthy or complex materials, aim for a high-level summary first, followed by more detailed points if required.

Fromat:
    subject: Few words to illustrate this point
    description: High level description of this point which shows the background, motivation, methodology and conclusion of this point demonstrated in this material
    importance_explanation: Why do we think this is an important point of this material? What's the role of this point playing in the structure of this material?
    technical_details: Provide data, approaches, formula or reference used in this material related to this point. The technical_details should be as detail as possible. Any number, term, forluma is appreciated.
    importance: How important is this point among all points? You can choose from 1,2,3,4,5. A large number means it's important. A small number means it's trivial.

            """


class StandardSummary(BaseSummary):
    """Standard summary type with motivation, conclusion and bullet points"""

    motivation: Motivation
    conclusion: Conclusion
    bullet_points: list[BulletPoint]

    @classmethod
    def get_rag_key_words(cls) -> list[str]:
        """Returns a list of key words important for RAG purposes."""
        base_keywords = super().get_rag_key_words()
        additional_keywords = [
            "conclusion",
            "findings",
            "results",
            "contribution",
            "solved",
            "outcome",
            "achievement",
            "impact",
            "significance",
            "future_work",
            "limitations",
            "recommendations",
        ]
        return base_keywords + additional_keywords

    def __str__(self):
        split_line = "=" * 20 + "\n"
        return f"""
        {split_line}
        motivation:{self.motivation}
        conclusion:{self.conclusion}
        bullet_points:{self.bullet_points}
        {split_line}
        """

    def model_dump(self):
        return {
            "summary_type": "standard",
            "motivation": self.motivation.model_dump(),
            "conclusion": self.conclusion.model_dump(),
            "bullet_points": [bp.model_dump() for bp in self.bullet_points],
        }

    @classmethod
    def excluded_fields(cls) -> list[str]:
        """Returns a list of fields that should be excluded during question set generation."""
        return ["summary_type"]

    @classmethod
    def prompt(cls):
        return """
I want you to act as a professional summarizer tasked with creating a comprehensive standard summary of the provided input material. This summary should include motivation, conclusion, and detailed bullet points. Follow these detailed instructions to ensure clarity and thoroughness:

Role and Objective
Role: Act as an expert summarizer with a focus on comprehensive analysis and structured presentation.
Objective: Produce a complete summary that includes motivation, conclusion, and well-organized bullet points that capture the key aspects of the material.

Structure Requirements:

1. Motivation Section:
   - Provide a clear description of the topic or problem
   - Explain the specific problem being addressed
   - Detail the proposed solution approach
   - Justify why the problem can be solved

2. Conclusion Section:
   - Summarize the main findings or outcomes
   - Restate the core problem addressed
   - Quantify or qualify how much of the problem was solved
   - Highlight the key contributions made

3. Bullet Points:
   - Break down the material into distinct, standalone concepts
   - Each point should be self-contained and clear
   - Include both high-level concepts and technical details
   - Prioritize points based on importance

Format for Each Bullet Point:
    subject: Brief, descriptive title for the point
    description: Comprehensive explanation including background, methodology, and findings
    importance_explanation: Justification of the point's significance in the overall context
    technical_details: Specific data, methods, formulas, or references used
    importance: Rating from 1-5 indicating relative significance (5 being most important)

Additional Guidelines:
- Maintain objectivity and avoid personal interpretations
- Use clear, precise language while preserving technical accuracy
- Ensure logical flow between sections
- Include relevant technical details while keeping explanations accessible
- Structure the content to build understanding progressively
"""


class TechnicalSummary(BaseModel):
    """Summary specifically for technical documentation"""

    overview: str
    key_concepts: list[str]
    technical_details: list[str]
    implementation_steps: list[str]
    requirements: list[str]
    limitations: list[str]

    @classmethod
    def get_rag_key_words(cls) -> list[str]:
        """Returns a list of key words important for RAG purposes."""
        return [
            "technical",
            "implementation",
            "architecture",
            "requirements",
            "specifications",
            "algorithms",
            "data_structures",
            "performance",
            "optimization",
            "maintenance",
            "deployment",
            "configuration",
            "dependencies",
            "limitations",
            "constraints",
            "metrics",
        ]

    def __str__(self):
        split_line = "=" * 20 + "\n"
        return f"""
        {split_line}
        overview:{self.overview}
        key_concepts:{self.key_concepts}
        technical_details:{self.technical_details}
        implementation_steps:{self.implementation_steps}
        requirements:{self.requirements}
        limitations:{self.limitations}
        {split_line}
        """

    def model_dump(self):
        return {
            "summary_type": "technical",
            "overview": self.overview,
            "key_concepts": self.key_concepts,
            "technical_details": self.technical_details,
            "implementation_steps": self.implementation_steps,
            "requirements": self.requirements,
            "limitations": self.limitations,
        }

    @classmethod
    def excluded_fields(cls) -> list[str]:
        """Returns a list of fields that should be excluded during question set generation."""
        return ["summary_type"]

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "title": "TechnicalSummary",
            "description": "Summary specifically for technical documentation",
            "properties": {
                "overview": {
                    "type": "string",
                    "description": "A high-level overview of the technical content",
                },
                "key_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key technical concepts",
                },
                "technical_details": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of technical details",
                },
                "implementation_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of implementation steps",
                },
                "requirements": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of requirements",
                },
                "limitations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of limitations",
                },
            },
            "required": [
                "overview",
                "key_concepts",
                "technical_details",
                "implementation_steps",
                "requirements",
                "limitations",
            ],
        }

    @classmethod
    def prompt(cls):
        return """
I want you to act as a technical documentation specialist tasked with creating a detailed technical summary. This summary should focus on the technical aspects, implementation details, and practical considerations of the subject matter. Follow these detailed instructions to ensure comprehensive technical coverage:

Role and Objective
Role: Act as a technical documentation expert with deep understanding of technical concepts and implementation details.
Objective: Produce a structured technical summary that covers all aspects of the technical documentation, from high-level concepts to specific implementation details.

Structure Requirements:

1. Overview:
   - Provide a concise technical overview of the system or concept
   - Focus on technical architecture and core functionality
   - Highlight the main technical objectives

2. Key Concepts:
   - List and explain fundamental technical concepts
   - Include relevant technical terminology
   - Explain relationships between different technical components

3. Technical Details:
   - Provide specific technical specifications
   - Include relevant formulas, algorithms, or methodologies
   - Detail technical parameters and configurations

4. Implementation Steps:
   - Break down the implementation process into clear, sequential steps
   - Include specific technical procedures
   - Provide guidance on technical setup and configuration

5. Requirements:
   - List all technical prerequisites
   - Specify system requirements
   - Detail dependencies and compatibility requirements

6. Limitations:
   - Document technical constraints
   - List known technical issues or limitations
   - Specify performance boundaries

Additional Guidelines:
- Use precise technical terminology
- Include specific version numbers, parameters, and configurations
- Provide clear technical specifications
- Include relevant code snippets or technical examples
- Maintain technical accuracy while ensuring clarity
- Consider both theoretical and practical technical aspects
"""


class InnovationSummary(BaseModel):
    """Summary specifically for innovation documentation"""

    overview: str
    key_concepts: list[str]
    innovation_points: list[InnovationPoint]
    references: list[str]

    @classmethod
    def get_rag_key_words(cls) -> list[str]:
        """Returns a list of key words important for RAG purposes."""
        return [
            "innovation",
            "novel",
            "breakthrough",
            "invention",
            "discovery",
            "advancement",
            "improvement",
            "enhancement",
            "originality",
            "creativity",
            "uniqueness",
            "research",
            "development",
            "prototype",
            "comparison",
            "performance",
            "efficiency",
            "effectiveness",
            "impact",
            "applications",
            "potential",
            "future_work",
            "challenges",
            "opportunities",
        ]

    def __str__(self):
        split_line = "=" * 20 + "\n"
        return f"""
        {split_line}
        overview:{self.overview}
        key_concepts:{self.key_concepts}
        innovation_points:{self.innovation_points}
        references:{self.references}
        {split_line}
        """

    def model_dump(self):
        return {
            "summary_type": "innovation",
            "overview": self.overview,
            "key_concepts": self.key_concepts,
            "innovation_points": [ip.model_dump() for ip in self.innovation_points],
            "references": self.references,
        }

    @classmethod
    def excluded_fields(cls) -> list[str]:
        """Returns a list of fields that should be excluded during question set generation."""
        return ["references", "summary_type"]

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "title": "InnovationSummary",
            "description": "Summary specifically for innovation documentation",
            "properties": {
                "overview": {
                    "type": "string",
                    "description": "A high-level overview of the innovation",
                },
                "key_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of key innovative concepts",
                },
                "innovation_points": {
                    "type": "array",
                    "items": InnovationPoint.model_json_schema(),
                    "description": "List of detailed innovation points",
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of references and related work",
                },
            },
            "required": [
                "overview",
                "key_concepts",
                "innovation_points",
                "references",
            ],
        }

    @classmethod
    def prompt(cls):
        return """
I want you to act as an innovation documentation specialist tasked with creating a detailed innovation summary. This summary should focus on identifying, analyzing, and explaining innovative aspects of the subject matter. Follow these detailed instructions to ensure comprehensive coverage of innovation:

Role and Objective
Role: Act as an innovation documentation expert with deep understanding of technological advancement and creative solutions.
Objective: Produce a structured innovation summary that highlights novel concepts, their significance, and their relationship to existing work.

Structure Requirements:

1. Overview:
   - Provide a concise overview of the innovative aspects
   - Highlight the main areas of innovation
   - Explain the broader impact and potential applications

2. Key Concepts:
   - List and explain fundamental innovative concepts
   - Identify breakthrough ideas and novel approaches
   - Highlight unique methodologies or techniques

3. Innovation Points:
   For each innovation point, provide:
   - subject: Brief, descriptive title of the innovation
   - description: Detailed explanation of the innovative concept
   - technical_details: Specific technical implementation or methodology
   - why_is_innovative: Clear explanation of what makes this concept novel
   - similar_concepts: List of related or similar existing concepts
   - importance: Rating from 1-5 indicating relative significance (5 being most important)

4. References:
   - List relevant academic papers and technical documentation
   - Include related work and prior art
   - Reference industry standards or frameworks

Additional Guidelines:
- Focus on truly innovative aspects rather than incremental improvements
- Clearly explain what makes each concept novel
- Provide concrete technical details to support innovation claims
- Compare and contrast with existing solutions
- Consider both theoretical and practical implications
- Maintain objectivity while highlighting unique contributions
- Include specific metrics or evidence of innovation where available
"""


class MetaDataSummary(BaseModel):
    """Summary specifically for metadata documentation"""

    paper_title: str
    authors: str
    journal_name: str
    publication_date: str

    @classmethod
    def get_rag_key_words(cls) -> list[str]:
        """Returns a list of key words important for RAG purposes."""
        return ["metadata", "title", "authors", "publication", "journal", "conference"]

    def __str__(self):
        split_line = "=" * 20 + "\n"
        return f"""
        {split_line}
        paper_title:{self.paper_title}
        authors:{self.authors}
        journal_name:{self.journal_name}
        publication_date:{self.publication_date}
        {split_line}
        """

    def model_dump(self):
        return {
            "summary_type": "metadata",
            "paper_title": self.paper_title,
            "authors": self.authors,
            "journal_name": self.journal_name,
            "publication_date": self.publication_date,
        }

    @classmethod
    def excluded_fields(cls) -> list[str]:
        """Returns a list of fields that should be excluded during question set generation."""
        return ["paper_title", "authors", "journal_name", "publication_date", "summary_type"]

    @classmethod
    def model_json_schema(cls):
        return {
            "type": "object",
            "title": "MetaDataSummary",
            "description": "Summary specifically for metadata documentation",
            "properties": {
                "paper_title": {
                    "type": "string",
                    "description": "Title of the paper or document",
                },
                "authors": {
                    "type": "string",
                    "description": "Authors of the paper or document",
                },
                "journal_name": {
                    "type": "string",
                    "description": "Name of the journal or publication venue",
                },
                "publication_date": {
                    "type": "string",
                    "description": "Date of publication",
                },
            },
            "required": [
                "paper_title",
                "authors",
                "journal_name",
                "publication_date",
            ],
        }

    @classmethod
    def prompt(cls):
        return """
I want you to act as a metadata documentation specialist tasked with creating a detailed metadata summary. This summary should focus on capturing the essential metadata information about a document or paper. Follow these detailed instructions to ensure comprehensive coverage:

Role and Objective
Role: Act as a metadata documentation expert with focus on accurate and complete metadata capture.
Objective: Produce a structured metadata summary that includes all relevant publication and authorship information.

Structure Requirements:

1. Paper Title:
   - Provide the complete and accurate title of the document
   - Include any subtitles or additional title information
   - Ensure proper formatting and capitalization

2. Authors:
   - List all authors in the correct order
   - Include full names and affiliations if available
   - Maintain proper formatting of author names

3. Journal Name:
   - Provide the complete name of the publication venue
   - Include any relevant volume or issue information
   - Specify the type of publication (journal, conference, etc.)

4. Publication Date:
   - Include the complete publication date
   - Use a consistent date format
   - Specify if the date is approximate or exact

Additional Guidelines:
- Ensure accuracy in all metadata fields
- Use consistent formatting throughout
- Include all available metadata information
- Maintain proper citation standards
- Verify the completeness of the information
"""
