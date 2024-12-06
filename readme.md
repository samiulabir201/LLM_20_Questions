# 20 Questions: A Language Model Simulation Project

## Overview
This project brings the classic deduction game **20 Questions** into the modern AI realm. The goal is to create language models (LLMs) capable of playing the game as a guesser or answerer in a strategic, cooperative, and competitive environment. The models are designed to demonstrate capabilities in logical deduction, efficient information retrieval, and collaborative gameplay.

**20 Questions** is a game where one player thinks of a word, and the other tries to guess it by asking up to 20 yes-or-no questions. The challenge lies in narrowing down possibilities through well-constructed queries and answers.

## Goals
1. **Showcase Advanced LLM Abilities**:
   - Demonstrate the ability of LLMs to ask insightful questions and reason effectively.
   - Enhance LLMs' capacity for understanding context and responding logically.

2. **Efficiency in Decision Making**:
   - Design strategies to minimize the number of questions needed to identify the target word.
   - Implement probabilistic methods and precomputed data to improve decision-making.

3. **Develop Generalizable AI Frameworks**:
   - Build reusable agents for answering and guessing in constrained environments.
   - Provide models tuned for specific tasks to enhance response accuracy.

4. **Real-World Applicability**:
   - Train LLMs to perform tasks requiring structured thinking and strategic questioning.
   - Explore the use of similar frameworks in problem-solving scenarios like customer support or medical diagnostics.

## Project Highlights
- **Keyword Database**: A carefully curated and ranked database of words with associated probabilities, enabling strategic question prioritization.
- **Specialized Agents**: Modular agents for answering and guessing, leveraging LLMs tuned for tasks like natural language processing and mathematical reasoning.
- **Custom Models**: Integration of models like Meta-Llama and DeepSeek-Math to enhance performance in specific question-answer scenarios.
- **Entropy Optimization**: Entropy-based question generation ensures the fastest narrowing of possible answers.

## How the Project Achieves Its Goals
1. **Logical Deduction Framework**:
   - Questions and answers are stored, analyzed, and used for progressive narrowing of possibilities.
   - Strategies are implemented for optimal guess selection using probabilistic methods.

2. **Task-Specific LLMs**:
   - A dedicated answerer handles queries with high accuracy.
   - The guesser uses entropy optimization to strategically narrow down the word list.

3. **High Performance through Preprocessing**:
   - Precomputed probabilities and keyword relationships enhance runtime efficiency.
   - Quantized models reduce computational load without sacrificing accuracy.

4. **Robust Testing and Validation**:
   - Extensive testing ensures compatibility with diverse scenarios and corner cases.
   - Debugging hooks make error detection and resolution intuitive.

## Real-World Contributions
- **Customer Support Systems**: The questioning framework can be applied to chatbots, helping them narrow down user issues more effectively.
- **Educational Tools**: Gamified AI assistants can engage students in logical thinking and decision-making.
- **Medical Diagnostics**: The structured deduction framework could assist doctors in identifying probable conditions through step-by-step questioning.
- **Collaborative AI Systems**: This project demonstrates how LLMs can work together, a concept applicable to multi-agent AI systems in complex environments.

## Getting Started
1. Clone the repository:
   git clone <repository-url>
   cd 20-questions-project
2. Install dependencies:
   pip install -r requirements.txt
3. Execute the main program:
   python src/main.py

For detailed usage and examples, refer to the documentation within the src folder.

## Technologies Used
1. Python
2. PyTorch
3. Hugging Face Transformers
4. Accelerate Library
5. BitsAndBytes Quantization

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.

## License
This project is licensed under the MIT License. See LICENSE for details.

MIT License

Copyright (c) 2024 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

