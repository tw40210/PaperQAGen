import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import PyPDF2
from pydantic import BaseModel

from src.qa_gpt.chat.chat import get_chat_gpt_response_structure_async
from src.qa_gpt.core.controller.rag_controller import RAGController
from src.qa_gpt.core.objects.questions import (
    MaterialClipsForTopic,
    MultipleChoiceQuestionSet,
)

T = TypeVar("T", bound=BaseModel)


class BaseQAController(ABC):
    @abstractmethod
    async def get_summary(
        self, file_id: str, summary_class: type[T], additional_context: str = ""
    ) -> T:
        """Get a summary of the content from a file.

        Args:
            file_id (str): ID of the file to summarize
            summary_class (Type[T]): The Pydantic model class to use for the summary
            additional_context (str): Additional context from markdown file, defaults to empty string

        Returns:
            T: A structured summary of the content
        """
        pass

    @abstractmethod
    async def get_questions(
        self, file_id: str, field_name: str, field_value: any, additional_context: str = ""
    ) -> MultipleChoiceQuestionSet:
        """Generate questions based on the content from a file and a specific field.

        Args:
            file_id (str): ID of the file to generate questions from
            field_name (str): Name of the field to generate questions for
            field_value (any): Value of the field to generate questions for
            additional_context (str): Additional context from markdown file, defaults to empty string

        Returns:
            MultipleChoiceQuestionSet: A set of questions for the specified field
        """
        pass

    @abstractmethod
    async def get_summaries_batch(
        self,
        file_ids: list[str],
        summary_classes: list[type[T]],
        additional_contexts: list[str] = None,
    ) -> list[T]:
        pass

    @abstractmethod
    async def get_questions_batch(
        self,
        file_ids: list[str],
        field_names: list[str],
        field_values: list[Any],
        additional_contexts: list[str] = None,
    ) -> list[MultipleChoiceQuestionSet]:
        pass


class QAController(BaseQAController):
    def __init__(self) -> None:
        self.preprocess_controller = PreprocessController()
        self.user_input_temp = {"role": "user", "content": "how can I solve 8x + 7 = -23"}
        self.summary_message_temp = {
            "role": "system",
            "content": """
Summary prompt placeholder
            """,
        }
        self.question_message_temp = {
            "role": "system",
            "content": """
I want you to act as a professional teacher and instructional designer tasked with creating insightful and comprehensive questions based on input materials provided in PDF format. The goal is to ensure students understand the material thoroughly and can demonstrate mastery of its core concepts. Follow the steps below:

IMPORTANT: You will be provided with carefully selected material clips that focus on a specific topic. These clips are the most relevant excerpts from the material. You MUST:
1. Base your questions ONLY on the information provided in these clips
2. Reference specific content from the clips in your questions and explanations
3. Ensure each question directly relates to the content in the clips
4. Use the clips as the sole source of information for creating questions

Role and Responsibilities
Role: Behave like an experienced teacher specializing in creating educational assessments.
Objective: Create questions that reflect the key ideas and concepts in the input materials and evaluate whether the students fully understand the content.
Steps to Follow
Analyze the Material:

Carefully review the input material, summarizing its main topics, subtopics, and key concepts.
Identify critical ideas, data points, theories, arguments, or processes mentioned in the material.
Note down any technical terms, definitions, formulas, or frameworks essential to understanding the content.
Organize Content:

Group related topics or sections to ensure logical question design.
Prioritize concepts based on their significance within the text and their complexity.
Determine Question Types:
Create a mix of question types to evaluate understanding at different cognitive levels:

Factual: Questions to confirm recall of critical facts, definitions, or processes.
Conceptual: Questions that test comprehension of broader ideas and relationships between concepts.
Application-Based: Questions requiring students to apply the concepts to new scenarios or problems.
Analytical: Questions encouraging students to evaluate arguments, compare theories, or draw conclusions.
Design Questions:

Write questions that are specific, concise, and directly related to the input material.
Avoid vague or overly general questions that could lead to superficial answers.
Ensure questions are challenging but fair, requiring a deep understanding of the content to answer correctly.
Include context, examples, or excerpts from the material if needed for clarity.
Check Coverage:

Cross-check the questions against the input material to ensure all critical concepts are addressed.
Make sure the questions collectively reflect the overall essence of the material.
Provide Correct Answers:

For each question, supply a correct and detailed answer or solution.
Explain the reasoning or logic behind the answer when applicable, especially for higher-order questions.
Iterate and Refine:

Review the created questions for clarity, relevance, and alignment with the material.
Make improvements to address any gaps or inconsistencies.
Additional Considerations
If the material is technical or includes jargon, include simpler questions to scaffold understanding for students new to the topic.
Ensure questions are culturally sensitive and accessible to the intended audience.

Format:
MultipleChoiceQuestion:
    question_description: The description of the question related to a bullet point to testify if a student fully understands this point. Must be based on the provided material clips.
    choice_1: A description can easily confuse students if it is true according to the question_description. Must be plausible based on the clips.
    choice_2: A description can easily confuse students if it is true according to the question_description. Must be plausible based on the clips.
    choice_3: A description can easily confuse students if it is true according to the question_description. Must be plausible based on the clips.
    choice_4: A description can easily confuse students if it is true according to the question_description. Must be plausible based on the clips.


Choice:
    choice_description: A description can easily confuse students if it is true according to the corresponding question_description. Must reference specific content from the clips.
    answer: If it is true or not, according to the corresponding question_description. This answer should be clear and well explanable.
    explanation: The reason why it is true or not, according to the corresponding question_description. Please provide as detail as possible to prove the answer, referencing specific information from the material clips.
            """,
        }
        self.material_clips_for_topic_temp = {
            "role": "system",
            "content": """
I want you to act as a professional content curator and educational specialist. Your task is to select the most relevant and informative clips from the provided material that best represent a specific topic. Follow these guidelines:

1. Clip Selection Criteria:
   - Select 3-5 most relevant clips that directly address the given topic
   - Each clip should be a complete, self-contained excerpt from the material
   - Clips should be verbatim from the material without any modifications
   - Clips should be of appropriate length (typically 1-3 sentences)
   - Ensure the clips collectively provide comprehensive coverage of the topic

2. For each selected clip:
   - Provide the exact text from the material
   - Explain why this clip was chosen, focusing on:
     * How it directly relates to the topic
     * What key information or insight it provides
     * Why it's essential for understanding the topic

3. Format your response as follows:
   Clip 1:
   - Text: [exact quote from material]
   - Reason: [detailed explanation of why this clip was chosen]

   Clip 2:
   - Text: [exact quote from material]
   - Reason: [detailed explanation of why this clip was chosen]

   [Continue for all selected clips]

4. Additional Guidelines:
   - Maintain the original context of each clip
   - Ensure the clips are exactly from the material without any modifications
   - Avoid selecting redundant or overlapping clips
   - Prioritize clips that contain unique or critical information
   - Consider the educational value and clarity of each clip

Remember: Your goal is to help learners understand the topic by providing the most relevant and informative excerpts from the material, along with clear explanations of their significance.
            """,
        }

    async def get_summary(
        self, file_id: str, summary_class: type[T], additional_context: str = ""
    ) -> T:
        rag_controller = RAGController.from_file_id(file_id)
        # Get relevant content using search_text
        relevant_content = rag_controller.search_text("summary", k=5)
        material_text = "\n".join([text for text, _ in relevant_content])

        user_input = self.user_input_temp.copy()
        context = f"{material_text}\n\nAdditional Context:\n{additional_context}"
        user_input.update({"content": context})
        sys_summary_message = self.summary_message_temp.copy()
        sys_summary_message.update({"content": summary_class.prompt()})

        messages = [sys_summary_message, user_input]
        result = await get_chat_gpt_response_structure_async(messages, res_obj=summary_class)
        return result

    async def get_material_clips_for_topic(self, file_id: str, topic: str) -> MaterialClipsForTopic:
        rag_controller = RAGController.from_file_id(file_id)
        # Get relevant content using search_text
        relevant_content = rag_controller.search_text(topic, k=5)
        material_text = "\n".join([text for text, _ in relevant_content])

        user_input = self.user_input_temp.copy()
        user_input.update({"content": f"Material: {material_text}\n\nTopic: {topic}"})
        sys_material_clips_for_topic_message = self.material_clips_for_topic_temp.copy()

        messages = [sys_material_clips_for_topic_message, user_input]
        result = await get_chat_gpt_response_structure_async(
            messages, res_obj=MaterialClipsForTopic
        )
        return result

    async def get_questions(
        self, file_id: str, field_name: str, field_value: any, additional_context: str = ""
    ) -> MultipleChoiceQuestionSet:
        """Generate questions based on the content from a file and a specific field.

        Args:
            file_id (str): ID of the file to generate questions from
            field_name (str): Name of the field to generate questions for
            field_value (any): Value of the field to generate questions for
            additional_context (str): Additional context from markdown file, defaults to empty string

        Returns:
            MultipleChoiceQuestionSet: A set of questions for the specified field
        """
        # Create a new RAGController instance for get_material_clips_for_topic
        # material_clips_for_topic = await self.get_material_clips_for_topic(file_id, field_value)

        # Create a new RAGController instance for the main question generation
        rag_controller = RAGController.from_file_id(file_id)
        relevant_content = rag_controller.search_text(field_name, k=5)
        material_text = "\n".join([text for text, _ in relevant_content])

        user_input = self.user_input_temp.copy()
        context = f"Material: \n\n{field_name.replace('_', ' ').title()}:\n{field_value}\n\nAdditional Context:\n{material_text}\n\n Additional; Context:\n{additional_context}"
        user_input.update({"content": context})
        messages = [self.question_message_temp.copy(), user_input]
        return await get_chat_gpt_response_structure_async(
            messages, res_obj=MultipleChoiceQuestionSet
        )

    async def get_summaries_batch(
        self,
        file_ids: list[str],
        summary_classes: list[type[T]],
        additional_contexts: list[str] = None,
    ) -> list[T]:
        tasks = []
        if additional_contexts is None:
            additional_contexts = [""] * len(file_ids)
        for file_id, summary_class, additional_context in zip(
            file_ids, summary_classes, additional_contexts
        ):
            tasks.append(self.get_summary(file_id, summary_class, additional_context))
            await asyncio.sleep(3)  # Add 3 seconds delay between calls
        return await asyncio.gather(*tasks)

    async def get_questions_batch(
        self,
        file_ids: list[str],
        field_names: list[str],
        field_values: list[Any],
        additional_contexts: list[str] = None,
    ) -> list[MultipleChoiceQuestionSet]:
        tasks = []
        if additional_contexts is None:
            additional_contexts = [""] * len(file_ids)
        for file_id, field_name, field_value, additional_context in zip(
            file_ids, field_names, field_values, additional_contexts
        ):
            tasks.append(self.get_questions(file_id, field_name, field_value, additional_context))
            await asyncio.sleep(3)  # Add 3 seconds delay between calls
        return await asyncio.gather(*tasks)


class PreprocessController:
    def __init__(self) -> None:
        pass

    def preprocess(self, file_path: Path):
        output = self._pdf_to_text(file_path)
        return output

    def _pdf_to_text(self, pdf_path: Path):
        # Open the PDF file in read-binary mode
        with open(str(pdf_path), "rb") as pdf_file:
            # Create a PdfReader object instead of PdfFileReader
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Initialize an empty string to store the text
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

        return text
